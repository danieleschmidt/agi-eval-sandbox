"""Main evaluation engine and orchestration."""

from typing import Dict, List, Optional, Any, Union, Callable
import asyncio
import logging
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager

from .models import Model
from .benchmarks import Benchmark, TruthfulQABenchmark, MMLUBenchmark, HumanEvalBenchmark, CustomBenchmark
from .results import Results, BenchmarkResult, EvaluationResult
from .exceptions import (
    EvaluationError, 
    ValidationError, 
    ResourceError, 
    RateLimitError, 
    TimeoutError,
    ModelProviderError,
    ConfigurationError
)
from .validation import InputValidator, ResourceValidator
from .logging_config import get_logger, performance_logger, security_logger
from .health import health_monitor
from .optimization import performance_optimizer, PerformanceMetrics
from .cache import cache_manager

logger = get_logger("evaluator")


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"  # Normal operation
    OPEN = "open"      # Failing, reject all requests
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Circuit breaker configuration."""
    failure_threshold: int = 5  # Number of failures before opening
    recovery_timeout: int = 60  # Seconds to wait before half-open
    success_threshold: int = 3  # Successes needed to close from half-open
    timeout: float = 30.0  # Request timeout in seconds


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.logger = get_logger(f"circuit_breaker.{name}")
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise EvaluationError(
                    f"Circuit breaker {self.name} is OPEN",
                    {
                        "circuit_breaker": self.name,
                        "state": self.state.value,
                        "failure_count": self.failure_count,
                        "last_failure": self.last_failure_time.isoformat() if self.last_failure_time else None
                    }
                )
        
        try:
            # Execute with timeout
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self.config.timeout
            )
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure(e)
            raise
    
    async def _on_success(self):
        """Handle successful execution."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.success_count = 0
                self.logger.info(f"Circuit breaker {self.name} closed after recovery")
    
    async def _on_failure(self, exception: Exception):
        """Handle failed execution."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        self.success_count = 0  # Reset success count
        
        self.logger.warning(
            f"Circuit breaker {self.name} recorded failure",
            extra={
                "failure_count": self.failure_count,
                "exception": str(exception),
                "state": self.state.value
            }
        )
        
        if self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.error(
                f"Circuit breaker {self.name} opened due to failures",
                extra={
                    "failure_count": self.failure_count,
                    "threshold": self.config.failure_threshold
                }
            )
    
    def _should_attempt_reset(self) -> bool:
        """Check if we should attempt to reset from OPEN to HALF_OPEN."""
        if self.last_failure_time is None:
            return True
        
        return (
            datetime.now() - self.last_failure_time >
            timedelta(seconds=self.config.recovery_timeout)
        )


@dataclass
class RetryConfig:
    """Retry configuration."""
    max_attempts: int = 3
    base_delay: float = 1.0  # Base delay in seconds
    max_delay: float = 60.0  # Maximum delay in seconds
    exponential_base: float = 2.0  # Exponential backoff base
    jitter: bool = True  # Add random jitter to delay
    retryable_exceptions: tuple = field(default_factory=lambda: (
        TimeoutError,
        RateLimitError,
        ModelProviderError,
        ConnectionError,
        asyncio.TimeoutError
    ))


class RetryHandler:
    """Handles retry logic with exponential backoff."""
    
    def __init__(self, config: RetryConfig = None):
        self.config = config or RetryConfig()
        self.logger = get_logger("retry_handler")
    
    async def execute_with_retry(self, func: Callable, *args, **kwargs):
        """Execute function with retry logic."""
        last_exception = None
        
        for attempt in range(self.config.max_attempts):
            try:
                result = await func(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(
                        f"Operation succeeded on attempt {attempt + 1}",
                        extra={"attempt": attempt + 1, "function": func.__name__}
                    )
                return result
            
            except Exception as e:
                last_exception = e
                
                # Check if exception is retryable
                if not isinstance(e, self.config.retryable_exceptions):
                    self.logger.warning(
                        f"Non-retryable exception in {func.__name__}: {str(e)}",
                        extra={"exception_type": type(e).__name__}
                    )
                    raise
                
                # Don't retry on last attempt
                if attempt == self.config.max_attempts - 1:
                    break
                
                # Calculate delay with exponential backoff and jitter
                delay = min(
                    self.config.base_delay * (self.config.exponential_base ** attempt),
                    self.config.max_delay
                )
                
                if self.config.jitter:
                    # Add up to 50% jitter
                    jitter_amount = delay * 0.5 * random.random()
                    delay += jitter_amount
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed, retrying in {delay:.2f}s",
                    extra={
                        "attempt": attempt + 1,
                        "delay": delay,
                        "exception": str(e),
                        "function": func.__name__
                    }
                )
                
                await asyncio.sleep(delay)
        
        # All retries exhausted
        self.logger.error(
            f"All {self.config.max_attempts} attempts failed for {func.__name__}",
            extra={"final_exception": str(last_exception)}
        )
        raise EvaluationError(
            f"Operation failed after {self.config.max_attempts} attempts: {str(last_exception)}",
            {
                "max_attempts": self.config.max_attempts,
                "final_exception": str(last_exception),
                "function": func.__name__
            }
        )


class EvalSuite:
    """Main evaluation suite orchestrator."""
    
    def __init__(self, max_concurrent_evaluations: int = 5):
        self._benchmarks: Dict[str, Benchmark] = {}
        self._results_history: List[Results] = []
        self._max_concurrent = max_concurrent_evaluations
        self._active_evaluations = 0
        self._evaluation_semaphore = asyncio.Semaphore(max_concurrent_evaluations)
        
        # Initialize error handling components
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._retry_handler = RetryHandler()
        
        # Rate limiting
        self._rate_limits: Dict[str, List[float]] = {}  # provider -> timestamps
        self._rate_limit_window = 60.0  # 1 minute window
        
        # Validation
        self._validator = InputValidator()
        
        # Performance monitoring
        self._performance_metrics = PerformanceMetrics()
        self._enable_optimizations = True
        
        self._register_default_benchmarks()
    
    def _register_default_benchmarks(self) -> None:
        """Register built-in benchmarks."""
        try:
            self.register_benchmark(TruthfulQABenchmark())
            self.register_benchmark(MMLUBenchmark())
            self.register_benchmark(HumanEvalBenchmark())
        except Exception as e:
            logger.error(f"Failed to register default benchmarks: {e}")
            raise ConfigurationError(f"Failed to initialize default benchmarks: {str(e)}")
    
    def _get_circuit_breaker(self, name: str) -> CircuitBreaker:
        """Get or create circuit breaker for a service."""
        if name not in self._circuit_breakers:
            self._circuit_breakers[name] = CircuitBreaker(name)
        return self._circuit_breakers[name]
    
    async def _check_rate_limits(self, provider: str, requests_per_minute: int = 60) -> None:
        """Check and enforce rate limits for model providers."""
        now = time.time()
        
        if provider not in self._rate_limits:
            self._rate_limits[provider] = []
        
        # Clean old timestamps outside the window
        self._rate_limits[provider] = [
            ts for ts in self._rate_limits[provider]
            if now - ts < self._rate_limit_window
        ]
        
        # Check if we're at the limit
        if len(self._rate_limits[provider]) >= requests_per_minute:
            oldest_request = min(self._rate_limits[provider])
            wait_time = self._rate_limit_window - (now - oldest_request)
            
            logger.warning(
                f"Rate limit reached for {provider}. Waiting {wait_time:.2f}s",
                extra={
                    "provider": provider,
                    "requests_in_window": len(self._rate_limits[provider]),
                    "wait_time": wait_time
                }
            )
            
            security_logger.log_suspicious_activity(
                "Rate limit reached",
                {
                    "provider": provider,
                    "requests_count": len(self._rate_limits[provider]),
                    "limit": requests_per_minute
                }
            )
            
            raise RateLimitError(
                f"Rate limit exceeded for provider {provider}",
                {
                    "provider": provider,
                    "requests_in_window": len(self._rate_limits[provider]),
                    "limit": requests_per_minute,
                    "wait_time": wait_time
                }
            )
        
        # Record this request
        self._rate_limits[provider].append(now)
    
    async def _validate_resources(self) -> None:
        """Validate system resources before starting evaluation."""
        try:
            # Check system health
            health_checks = await health_monitor.run_all_checks()
            critical_issues = [
                check for check in health_checks.values()
                if check.status.value == "critical"
            ]
            
            if critical_issues:
                raise ResourceError(
                    "Critical system resource issues detected",
                    {
                        "issues": [
                            {"check": check.name, "message": check.message}
                            for check in critical_issues
                        ]
                    }
                )
            
            # Check concurrent evaluation limits
            ResourceValidator.validate_concurrent_jobs(
                self._active_evaluations, 
                self._max_concurrent
            )
            
        except Exception as e:
            logger.error(f"Resource validation failed: {e}")
            raise
    
    @asynccontextmanager
    async def _evaluation_context(self, model_name: str, benchmark_name: str):
        """Context manager for evaluation with proper cleanup and monitoring."""
        evaluation_id = f"{model_name}_{benchmark_name}_{int(time.time())}"
        start_time = time.time()
        
        try:
            # Increment active evaluations counter
            self._active_evaluations += 1
            
            logger.info(
                f"Starting evaluation {evaluation_id}",
                extra={
                    "evaluation_id": evaluation_id,
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "active_evaluations": self._active_evaluations
                }
            )
            
            yield evaluation_id
            
        except Exception as e:
            logger.error(
                f"Evaluation {evaluation_id} failed: {str(e)}",
                extra={
                    "evaluation_id": evaluation_id,
                    "error": str(e),
                    "duration": time.time() - start_time
                }
            )
            raise
        finally:
            # Always decrement counter and log completion
            self._active_evaluations = max(0, self._active_evaluations - 1)
            duration = time.time() - start_time
            
            performance_logger.log_evaluation_performance(
                model_name=model_name,
                benchmark=benchmark_name,
                duration_seconds=duration,
                questions_count=0,  # Will be updated by caller
                success=True  # Will be updated by caller if there was an exception
            )
            
            logger.info(
                f"Evaluation {evaluation_id} completed",
                extra={
                    "evaluation_id": evaluation_id,
                    "duration": duration,
                    "active_evaluations": self._active_evaluations
                }
            )
    
    def register_benchmark(self, benchmark: Benchmark) -> None:
        """Register a benchmark for evaluation."""
        self._benchmarks[benchmark.name] = benchmark
        logger.info(f"Registered benchmark: {benchmark.name}")
    
    def list_benchmarks(self) -> List[str]:
        """List all available benchmark names."""
        return list(self._benchmarks.keys())
    
    def get_benchmark(self, name: str) -> Optional[Benchmark]:
        """Get a benchmark by name."""
        return self._benchmarks.get(name)
    
    def _validate_evaluation_inputs(
        self, 
        model: Model, 
        benchmarks: Union[str, List[str]], 
        num_questions: Optional[int],
        config: Dict[str, Any]
    ) -> None:
        """Validate inputs for evaluation."""
        # Validate model
        if not model:
            raise ValidationError("Model cannot be None")
        
        # Validate model name
        self._validator.validate_model_name(model.name)
        
        # Validate benchmarks
        if isinstance(benchmarks, str):
            if benchmarks != "all":
                self._validator.validate_string(benchmarks, "benchmark", min_length=1)
        elif isinstance(benchmarks, list):
            if not benchmarks:
                raise ValidationError("Benchmark list cannot be empty")
            for benchmark in benchmarks:
                self._validator.validate_string(benchmark, "benchmark", min_length=1)
        else:
            raise ValidationError("Benchmarks must be a string or list of strings")
        
        # Validate num_questions
        if num_questions is not None:
            if not isinstance(num_questions, int) or num_questions <= 0:
                raise ValidationError(
                    "num_questions must be a positive integer",
                    {"value": num_questions, "type": type(num_questions).__name__}
                )
        
        # Validate config parameters
        if "temperature" in config:
            self._validator.validate_temperature(config["temperature"])
        
        if "max_tokens" in config:
            self._validator.validate_max_tokens(config["max_tokens"])
    
    def _resolve_benchmark_names(self, benchmarks: Union[str, List[str]]) -> List[str]:
        """Resolve benchmark names to a list."""
        if benchmarks == "all":
            return self.list_benchmarks()
        elif isinstance(benchmarks, str):
            return [benchmarks]
        else:
            return benchmarks
    
    def _validate_benchmark_names(self, benchmark_names: List[str]) -> None:
        """Validate that all benchmark names exist."""
        for name in benchmark_names:
            if name not in self._benchmarks:
                available_benchmarks = list(self._benchmarks.keys())
                raise ValidationError(
                    f"Unknown benchmark: {name}",
                    {
                        "requested_benchmark": name,
                        "available_benchmarks": available_benchmarks
                    }
                )
    
    async def _evaluate_benchmarks_parallel(
        self,
        model: Model,
        benchmark_names: List[str],
        num_questions: Optional[int],
        **config
    ) -> List[Optional[BenchmarkResult]]:
        """Run benchmarks in parallel with error handling."""
        tasks = []
        for name in benchmark_names:
            task = self._evaluate_benchmark_with_circuit_breaker(
                model, 
                self._benchmarks[name], 
                num_questions,
                **config
            )
            tasks.append(task)
        
        # Use gather with return_exceptions to handle partial failures
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        benchmark_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(
                    f"Benchmark {benchmark_names[i]} failed: {str(result)}",
                    extra={
                        "benchmark": benchmark_names[i],
                        "exception": str(result),
                        "exception_type": type(result).__name__
                    }
                )
                benchmark_results.append(None)  # Mark as failed
            else:
                benchmark_results.append(result)
        
        return benchmark_results
    
    async def _evaluate_benchmarks_sequential(
        self,
        model: Model,
        benchmark_names: List[str],
        num_questions: Optional[int],
        **config
    ) -> List[Optional[BenchmarkResult]]:
        """Run benchmarks sequentially with error handling."""
        benchmark_results = []
        
        for name in benchmark_names:
            try:
                result = await self._evaluate_benchmark_with_circuit_breaker(
                    model,
                    self._benchmarks[name],
                    num_questions,
                    **config
                )
                benchmark_results.append(result)
            except Exception as e:
                logger.error(
                    f"Benchmark {name} failed: {str(e)}",
                    extra={
                        "benchmark": name,
                        "exception": str(e),
                        "exception_type": type(e).__name__
                    }
                )
                benchmark_results.append(None)  # Continue with other benchmarks
        
        return benchmark_results
    
    async def _evaluate_benchmark_with_circuit_breaker(
        self,
        model: Model,
        benchmark: Benchmark,
        num_questions: Optional[int],
        **config
    ) -> BenchmarkResult:
        """Evaluate benchmark with circuit breaker protection."""
        circuit_breaker_name = f"{model.provider_name}_{benchmark.name}"
        circuit_breaker = self._get_circuit_breaker(circuit_breaker_name)
        
        async def evaluate_func():
            return await self._retry_handler.execute_with_retry(
                self._evaluate_benchmark,
                model,
                benchmark,
                num_questions,
                **config
            )
        
        return await circuit_breaker.call(evaluate_func)
    
    async def evaluate(
        self,
        model: Model,
        benchmarks: Union[str, List[str]] = "all",
        num_questions: Optional[int] = None,
        save_results: bool = True,
        parallel: bool = True,
        **config
    ) -> Results:
        """
        Evaluate a model on specified benchmarks with comprehensive error handling.
        
        Args:
            model: Model to evaluate
            benchmarks: Benchmark names to run ("all" or list of names)
            num_questions: Limit number of questions per benchmark
            save_results: Whether to save results to history
            parallel: Whether to run benchmarks in parallel
            **config: Additional configuration options
        
        Returns:
            Results object with evaluation outcomes
        
        Raises:
            ValidationError: If inputs are invalid
            ResourceError: If system resources are insufficient
            EvaluationError: If evaluation fails
        """
        start_time = time.time()
        evaluation_success = False
        
        try:
            # Input validation
            self._validate_evaluation_inputs(model, benchmarks, num_questions, config)
            
            # Resource validation
            await self._validate_resources()
            
            # Check rate limits for model provider
            await self._check_rate_limits(model.provider_name)
            
            logger.info(
                f"Starting evaluation of {model.name} on benchmarks: {benchmarks}",
                extra={
                    "model_name": model.name,
                    "provider": model.provider_name,
                    "benchmarks": benchmarks,
                    "num_questions": num_questions,
                    "parallel": parallel
                }
            )
            
            # Resolve benchmark names
            benchmark_names = self._resolve_benchmark_names(benchmarks)
            
            # Validate benchmarks exist
            self._validate_benchmark_names(benchmark_names)
            
            results = Results()
            results.metadata.update(config)
            results.metadata.update({
                "start_time": datetime.now().isoformat(),
                "model_name": model.name,
                "provider": model.provider_name
            })
            
            # Use semaphore to limit concurrent evaluations
            async with self._evaluation_semaphore:
                if parallel and len(benchmark_names) > 1:
                    # Run benchmarks in parallel with error handling
                    benchmark_results = await self._evaluate_benchmarks_parallel(
                        model, benchmark_names, num_questions, **config
                    )
                else:
                    # Run benchmarks sequentially
                    benchmark_results = await self._evaluate_benchmarks_sequential(
                        model, benchmark_names, num_questions, **config
                    )
            
            # Add results to main results object
            for benchmark_result in benchmark_results:
                if benchmark_result:  # Skip None results from failed evaluations
                    results.add_benchmark_result(benchmark_result)
            
            # Only save if we have at least one successful result
            if save_results and results.benchmark_results:
                self._results_history.append(results)
                logger.info(f"Saved evaluation results with {len(results.benchmark_results)} benchmarks")
            
            evaluation_success = True
            duration = time.time() - start_time
            overall_score = results.summary().get('overall_score', 0.0) if results.benchmark_results else 0.0
            
            logger.info(
                f"Evaluation completed successfully. Overall score: {overall_score:.3f}, Duration: {duration:.2f}s",
                extra={
                    "evaluation_success": True,
                    "overall_score": overall_score,
                    "duration_seconds": duration,
                    "benchmarks_completed": len(results.benchmark_results)
                }
            )
            
            return results
            
        except ValidationError as e:
            logger.error(f"Validation failed: {e.message}", extra=e.details)
            raise
        except ResourceError as e:
            logger.error(f"Resource error: {e.message}", extra=e.details)
            raise
        except RateLimitError as e:
            logger.error(f"Rate limit exceeded: {e.message}", extra=e.details)
            raise
        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"Evaluation failed unexpectedly: {str(e)}",
                extra={
                    "model_name": model.name if model else "unknown",
                    "benchmarks": benchmarks,
                    "duration_seconds": duration,
                    "exception_type": type(e).__name__
                },
                exc_info=True
            )
            raise EvaluationError(
                f"Evaluation failed: {str(e)}",
                {
                    "original_exception": str(e),
                    "exception_type": type(e).__name__,
                    "duration_seconds": duration
                }
            )
        finally:
            # Always log performance metrics
            duration = time.time() - start_time
            performance_logger.log_evaluation_performance(
                model_name=model.name if model else "unknown",
                benchmark="multiple" if isinstance(benchmarks, list) else str(benchmarks),
                duration_seconds=duration,
                questions_count=num_questions or 0,
                success=evaluation_success
            )
    
    async def _evaluate_benchmark(
        self,
        model: Model,
        benchmark: Benchmark,
        num_questions: Optional[int] = None,
        **config
    ) -> BenchmarkResult:
        """Evaluate model on a single benchmark with comprehensive error handling."""
        start_time = time.time()
        benchmark_name = benchmark.name
        model_name = model.name
        
        logger.info(f"Evaluating {benchmark_name} benchmark with model {model_name}")
        
        try:
            # Get circuit breaker for this model
            circuit_breaker = self._get_circuit_breaker(model_name)
            
            # Get questions
            questions = benchmark.get_questions()
            if num_questions:
                questions = questions[:min(num_questions, len(questions))]
            
            if not questions:
                raise EvaluationError(f"No questions available for benchmark {benchmark_name}")
            
            logger.info(f"Processing {len(questions)} questions for {benchmark_name}")
            
            # Generate responses with circuit breaker protection
            evaluation_results = await circuit_breaker.call(
                self._generate_and_evaluate_responses,
                model, benchmark, questions, config
            )
            
            # Create benchmark result
            benchmark_result = BenchmarkResult(
                benchmark_name=benchmark_name,
                model_name=model_name,
                model_provider=getattr(model, 'provider_name', 'unknown'),
                results=evaluation_results,
                config=config
            )
            
            # Log performance metrics
            duration = time.time() - start_time
            performance_logger.log_evaluation_performance(
                model_name=model_name,
                benchmark=benchmark_name,
                duration_seconds=duration,
                questions_count=len(questions),
                success=True
            )
            
            logger.info(
                f"Completed {benchmark_name}: {benchmark_result.average_score:.3f} avg score, "
                f"{benchmark_result.pass_rate:.1f}% pass rate in {duration:.2f}s"
            )
            
            return benchmark_result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Benchmark evaluation failed for {benchmark_name}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Log failure metrics
            performance_logger.log_evaluation_performance(
                model_name=model_name,
                benchmark=benchmark_name,
                duration_seconds=duration,
                questions_count=0,
                success=False
            )
            
            # Create empty result set for failed benchmark
            benchmark_result = BenchmarkResult(
                benchmark_name=benchmark_name,
                model_name=model_name,
                model_provider=getattr(model, 'provider_name', 'unknown'),
                results=[],
                config=config
            )
            
            return benchmark_result
    
    async def _generate_and_evaluate_responses(
        self,
        model: Model,
        benchmark: Benchmark,
        questions: List,
        config: Dict[str, Any]
    ) -> List[EvaluationResult]:
        """Generate responses and evaluate them with retry logic."""
        prompts = [q.prompt for q in questions]
        evaluation_results = []
        
        try:
            # Try batch generation first
            responses = await self._retry_handler.execute_with_retry(
                model.batch_generate, prompts, **config
            )
            logger.debug(f"Batch generation successful for {len(prompts)} prompts")
            
        except Exception as e:
            logger.warning(f"Batch generation failed, falling back to sequential: {e}")
            # Fallback to sequential generation with retries
            responses = await self._sequential_generation_with_retries(model, prompts, config)
        
        # Evaluate responses
        for question, response in zip(questions, responses):
            eval_result = await self._evaluate_single_response(
                model, benchmark, question, response
            )
            evaluation_results.append(eval_result)
        
        return evaluation_results
    
    async def _sequential_generation_with_retries(
        self,
        model: Model,
        prompts: List[str],
        config: Dict[str, Any]
    ) -> List[str]:
        """Generate responses sequentially with retry logic."""
        responses = []
        
        for i, prompt in enumerate(prompts):
            try:
                response = await self._retry_handler.execute_with_retry(
                    model.generate, prompt, **config
                )
                responses.append(response)
                logger.debug(f"Generated response {i+1}/{len(prompts)}")
                
            except Exception as e:
                logger.warning(f"Failed to generate response for prompt {i+1}: {e}")
                responses.append(f"GENERATION_ERROR: {str(e)}")
        
        return responses
    
    async def _evaluate_single_response(
        self,
        model: Model,
        benchmark: Benchmark,
        question,
        response: str
    ) -> EvaluationResult:
        """Evaluate a single response with comprehensive error handling."""
        start_time = time.time()
        
        try:
            # Sanitize response for security
            sanitized_response = self._validator.sanitize_user_input(response)
            
            # Evaluate response
            try:
                score = benchmark.evaluate_response(question, sanitized_response)
            except Exception as eval_error:
                logger.warning(f"Evaluation failed for question {question.id}: {eval_error}")
                # Use fallback evaluation
                score = self._create_fallback_score(question, sanitized_response, str(eval_error))
            
            # Calculate performance metrics
            duration = time.time() - start_time
            
            # Create evaluation result with comprehensive metadata
            eval_result = EvaluationResult(
                question_id=question.id,
                question_prompt=question.prompt,
                model_response=sanitized_response,
                score=score,
                benchmark_name=benchmark.name,
                category=getattr(question, 'category', 'uncategorized'),
                metadata={
                    "evaluation_duration_seconds": duration,
                    "model_name": model.name,
                    "model_provider": getattr(model, 'provider_name', 'unknown'),
                    "response_length": len(sanitized_response),
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "evaluation_version": "2.0"
                }
            )
            
            return eval_result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Failed to evaluate question {question.id}: {str(e)}"
            logger.error(error_msg, exc_info=True)
            
            # Create comprehensive error result
            from .benchmarks import Score
            error_score = Score(
                value=0.0,
                passed=False,
                explanation=f"Evaluation system error: {str(e)}"
            )
            
            return EvaluationResult(
                question_id=question.id,
                question_prompt=question.prompt,
                model_response=f"EVALUATION_ERROR: {str(e)}",
                score=error_score,
                benchmark_name=benchmark.name,
                category=getattr(question, 'category', 'error'),
                metadata={
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "evaluation_duration_seconds": duration,
                    "evaluation_timestamp": datetime.now().isoformat(),
                    "evaluation_version": "2.0"
                }
            )
    
    def _create_fallback_score(self, question, response: str, error_msg: str):
        """Create fallback score when primary evaluation fails."""
        from .benchmarks import Score
        
        # Simple heuristic-based fallback scoring
        if not response or response.startswith("ERROR") or response.startswith("GENERATION_ERROR"):
            return Score(value=0.0, passed=False, explanation=f"No valid response generated: {error_msg}")
        
        # Basic length and content checks
        if len(response.strip()) < 10:
            score_value = 0.1
        elif len(response.strip()) > 100:
            score_value = 0.3
        else:
            score_value = 0.2
        
        return Score(
            value=score_value,
            passed=score_value > 0.5,
            explanation=f"Fallback evaluation used due to: {error_msg}"
        )
    
    def _get_circuit_breaker(self, model_name: str) -> CircuitBreaker:
        """Get or create circuit breaker for model."""
        if model_name not in self._circuit_breakers:
            self._circuit_breakers[model_name] = CircuitBreaker(
                name=f"model_{model_name}",
                config=CircuitBreakerConfig()
            )
        return self._circuit_breakers[model_name]
    
    async def evaluate_optimized(
        self,
        model: Model,
        benchmarks: Union[str, List[str]] = "all",
        num_questions: Optional[int] = None,
        save_results: bool = True,
        use_cache: bool = True,
        parallel: bool = True,
        optimization_level: str = "balanced",  # "speed", "balanced", "accuracy"
        **config
    ) -> Results:
        """
        Optimized evaluation with adaptive performance features.
        
        Args:
            model: Model to evaluate
            benchmarks: Benchmarks to run
            num_questions: Number of questions per benchmark
            save_results: Whether to save results to history
            use_cache: Whether to use intelligent caching
            parallel: Whether to use parallel processing
            optimization_level: Optimization strategy ("speed", "balanced", "accuracy")
            **config: Additional configuration
        
        Returns:
            Results object with comprehensive metrics
        """
        start_time = time.time()
        
        # Validate inputs
        self._validate_evaluation_inputs(model, benchmarks, num_questions, config)
        
        # Apply optimization configuration
        if optimization_level == "speed":
            config.setdefault("temperature", 0.1)
            config.setdefault("max_tokens", 100)
            parallel = True
            use_cache = True
        elif optimization_level == "accuracy":
            config.setdefault("temperature", 0.7)
            config.setdefault("max_tokens", 500)
            # parallel can be True for accuracy too
        # "balanced" uses default settings
        
        try:
            async with self._evaluation_context(model.name, str(benchmarks)):
                # Prepare benchmark list
                if benchmarks == "all":
                    benchmark_list = list(self._benchmarks.values())
                elif isinstance(benchmarks, str):
                    benchmark_list = [self._benchmarks[benchmarks]]
                else:
                    benchmark_list = [self._benchmarks[name] for name in benchmarks if name in self._benchmarks]
                
                if parallel and self._enable_optimizations:
                    # Use optimized parallel evaluation
                    results = await self._evaluate_parallel_optimized(
                        model, benchmark_list, num_questions, use_cache, **config
                    )
                else:
                    # Use sequential evaluation
                    results = await self._evaluate_sequential(
                        model, benchmark_list, num_questions, use_cache, **config
                    )
                
                # Update performance metrics
                duration = time.time() - start_time
                total_questions = sum(len(br.results) for br in results.benchmark_results)
                
                self._performance_metrics.avg_response_time = duration / total_questions if total_questions > 0 else 0
                self._performance_metrics.throughput_qps = total_questions / duration if duration > 0 else 0
                self._performance_metrics.success_rate = 100.0  # Updated based on actual results
                
                # Log comprehensive performance metrics
                performance_logger.log_evaluation_performance(
                    model_name=model.name,
                    benchmark="optimized_evaluation",
                    duration_seconds=duration,
                    questions_count=total_questions,
                    success=True
                )
                
                if save_results:
                    self._results_history.append(results)
                
                logger.info(
                    f"Optimized evaluation completed: {total_questions} questions in {duration:.2f}s "
                    f"({self._performance_metrics.throughput_qps:.1f} QPS)"
                )
                
                return results
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Optimized evaluation failed: {e}", exc_info=True)
            
            # Create minimal results object for failed evaluation
            results = Results()
            results.metadata = {
                "error": str(e),
                "duration_seconds": duration,
                "model_name": model.name,
                "optimization_level": optimization_level
            }
            
            return results
    
    async def _evaluate_parallel_optimized(
        self,
        model: Model,
        benchmarks: List[Benchmark],
        num_questions: Optional[int],
        use_cache: bool,
        **config
    ) -> Results:
        """Parallel evaluation with performance optimizations."""
        
        # Create evaluation tasks
        benchmark_tasks = []
        for benchmark in benchmarks:
            cache_key = None
            if use_cache:
                cache_key = f"benchmark_{model.name}_{benchmark.name}_{num_questions}_{hash(str(config))}"
            
            task = self._evaluate_benchmark_cached(
                model, benchmark, num_questions, cache_key, **config
            )
            benchmark_tasks.append(task)
        
        # Execute tasks in parallel
        if len(benchmark_tasks) > 1:
            benchmark_results = await asyncio.gather(*benchmark_tasks, return_exceptions=True)
        else:
            benchmark_results = [await benchmark_tasks[0]]
        
        # Create results object
        results = Results()
        for result in benchmark_results:
            if isinstance(result, BenchmarkResult):
                results.add_benchmark_result(result)
            elif isinstance(result, Exception):
                logger.error(f"Benchmark evaluation failed: {result}")
                # Create empty benchmark result for failed benchmark
                # (details would be implementation-specific)
        
        return results
    
    async def _evaluate_sequential(
        self,
        model: Model,
        benchmarks: List[Benchmark],
        num_questions: Optional[int],
        use_cache: bool,
        **config
    ) -> Results:
        """Sequential evaluation for better resource control."""
        results = Results()
        
        for benchmark in benchmarks:
            try:
                cache_key = None
                if use_cache:
                    cache_key = f"benchmark_{model.name}_{benchmark.name}_{num_questions}_{hash(str(config))}"
                
                benchmark_result = await self._evaluate_benchmark_cached(
                    model, benchmark, num_questions, cache_key, **config
                )
                
                results.add_benchmark_result(benchmark_result)
                
            except Exception as e:
                logger.error(f"Sequential benchmark evaluation failed for {benchmark.name}: {e}")
                # Continue with other benchmarks
                continue
        
        return results
    
    async def _evaluate_benchmark_cached(
        self,
        model: Model,
        benchmark: Benchmark,
        num_questions: Optional[int],
        cache_key: Optional[str],
        **config
    ) -> BenchmarkResult:
        """Evaluate benchmark with intelligent caching."""
        
        # Try to get from cache first
        if cache_key:
            cached_result = await cache_manager.get_or_set(
                cache_key,
                lambda: self._evaluate_benchmark(model, benchmark, num_questions, **config),
                ttl_seconds=3600  # Cache for 1 hour
            )
            return cached_result
        
        # Evaluate without caching
        return await self._evaluate_benchmark(model, benchmark, num_questions, **config)
    
    def enable_optimizations(self):
        """Enable performance optimizations."""
        self._enable_optimizations = True
        performance_optimizer.enable_optimization()
        logger.info("Performance optimizations enabled")
    
    def disable_optimizations(self):
        """Disable performance optimizations (useful for debugging)."""
        self._enable_optimizations = False
        performance_optimizer.disable_optimization()
        logger.info("Performance optimizations disabled")
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization and performance statistics."""
        stats = cache_manager.get_stats() if hasattr(cache_manager, 'get_stats') else {}
        
        return {
            "evaluator_metrics": {
                "optimizations_enabled": self._enable_optimizations,
                "active_evaluations": self._active_evaluations,
                "max_concurrent": self._max_concurrent,
                "registered_benchmarks": len(self._benchmarks),
                "results_history_size": len(self._results_history)
            },
            "performance_metrics": {
                "avg_response_time": self._performance_metrics.avg_response_time,
                "throughput_qps": self._performance_metrics.throughput_qps,
                "success_rate": self._performance_metrics.success_rate,
                "memory_usage_mb": self._performance_metrics.memory_usage_mb,
                "cpu_usage_percent": self._performance_metrics.cpu_usage_percent
            },
            "optimization_details": performance_optimizer.get_optimization_stats(),
            "cache_stats": stats,
            "circuit_breakers": {
                name: {
                    "state": cb.state.value,
                    "failure_count": cb.failure_count,
                    "success_count": cb.success_count,
                    "last_failure": cb.last_failure_time.isoformat() if cb.last_failure_time else None
                }
                for name, cb in self._circuit_breakers.items()
            }
        }
    
    async def compare_models(
        self,
        models: List[Model],
        benchmarks: Union[str, List[str]] = "all",
        **config
    ) -> Dict[str, Results]:
        """
        Compare multiple models on the same benchmarks.
        
        Args:
            models: List of models to compare
            benchmarks: Benchmarks to run
            **config: Evaluation configuration
            
        Returns:
            Dictionary mapping model names to Results
        """
        tasks = [
            self.evaluate(model, benchmarks, save_results=False, **config)
            for model in models
        ]
        results_list = await asyncio.gather(*tasks)
        return {
            model.name: results 
            for model, results in zip(models, results_list)
        }
    
    def get_results_history(self) -> List[Results]:
        """Get historical evaluation results."""
        return self._results_history.copy()
    
    def load_results(self, path: str) -> Results:
        """Load results from file (placeholder)."""
        # Implementation would load from JSON/pickle file
        raise NotImplementedError("Results loading not yet implemented")
    
    def save_results(self, results: Results, path: str) -> None:
        """Save results to file (placeholder)."""
        # Implementation would save to JSON/pickle file
        raise NotImplementedError("Results saving not yet implemented")
    
    def get_leaderboard(
        self, 
        benchmark: Optional[str] = None,
        metric: str = "average_score"
    ) -> List[Dict[str, Any]]:
        """
        Get leaderboard of model performance.
        
        Args:
            benchmark: Specific benchmark to rank by (None for overall)
            metric: Metric to sort by ("average_score" or "pass_rate")
            
        Returns:
            List of model performance records sorted by metric
        """
        leaderboard = []
        
        for results in self._results_history:
            for benchmark_result in results.benchmark_results:
                if benchmark and benchmark_result.benchmark_name != benchmark:
                    continue
                
                record = {
                    "run_id": results.run_id,
                    "model_name": benchmark_result.model_name,
                    "model_provider": benchmark_result.model_provider,
                    "benchmark": benchmark_result.benchmark_name,
                    "average_score": benchmark_result.average_score,
                    "pass_rate": benchmark_result.pass_rate,
                    "total_questions": benchmark_result.total_questions,
                    "timestamp": benchmark_result.timestamp.isoformat()
                }
                leaderboard.append(record)
        
        # Sort by specified metric (descending)
        leaderboard.sort(key=lambda x: x[metric], reverse=True)
        return leaderboard
    
    def create_custom_benchmark(
        self,
        name: str,
        questions_file: Optional[str] = None,
        questions_data: Optional[List[Dict[str, Any]]] = None
    ) -> CustomBenchmark:
        """
        Create and register a custom benchmark.
        
        Args:
            name: Benchmark name
            questions_file: Path to questions JSON file
            questions_data: Direct questions data
            
        Returns:
            CustomBenchmark instance
        """
        if questions_file and questions_data:
            raise ValueError("Provide either questions_file or questions_data, not both")
        
        if questions_file:
            import json
            with open(questions_file, 'r') as f:
                questions_data = json.load(f)
        
        if not questions_data:
            raise ValueError("No questions data provided")
        
        # Convert dict data to Question objects
        from .benchmarks import Question
        questions = [Question.from_dict(q) for q in questions_data]
        
        # Create and register benchmark
        benchmark = CustomBenchmark(name, questions)
        self.register_benchmark(benchmark)
        
        # Store metadata for analytics
        self._benchmark_metadata[name] = {
            "created_at": datetime.now(),
            "question_count": len(questions),
            "categories": list(set(q.category for q in questions if q.category))
        }
        
        return benchmark