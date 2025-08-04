"""Main evaluation engine and orchestration."""

from typing import Dict, List, Optional, Any, Union
import asyncio
import logging
from datetime import datetime
from pathlib import Path

from .models import Model
from .benchmarks import Benchmark, TruthfulQABenchmark, MMLUBenchmark, HumanEvalBenchmark, CustomBenchmark
from .results import Results, BenchmarkResult, EvaluationResult


logger = logging.getLogger(__name__)


class EvalSuite:
    """Main evaluation suite orchestrator."""
    
    def __init__(self):
        self._benchmarks: Dict[str, Benchmark] = {}
        self._results_history: List[Results] = []
        self._register_default_benchmarks()
    
    def _register_default_benchmarks(self) -> None:
        """Register built-in benchmarks."""
        self.register_benchmark(TruthfulQABenchmark())
        self.register_benchmark(MMLUBenchmark())
        self.register_benchmark(HumanEvalBenchmark())
    
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
        Evaluate a model on specified benchmarks.
        
        Args:
            model: Model to evaluate
            benchmarks: Benchmark names to run ("all" or list of names)
            num_questions: Limit number of questions per benchmark
            save_results: Whether to save results to history
            parallel: Whether to run benchmarks in parallel
            **config: Additional configuration options
        
        Returns:
            Results object with evaluation outcomes
        """
        logger.info(f"Starting evaluation of {model.name} on benchmarks: {benchmarks}")
        
        # Resolve benchmark names
        if benchmarks == "all":
            benchmark_names = self.list_benchmarks()
        elif isinstance(benchmarks, str):
            benchmark_names = [benchmarks]
        else:
            benchmark_names = benchmarks
        
        # Validate benchmarks exist
        for name in benchmark_names:
            if name not in self._benchmarks:
                raise ValueError(f"Unknown benchmark: {name}")
        
        results = Results()
        results.metadata.update(config)
        
        if parallel:
            # Run benchmarks in parallel
            tasks = [
                self._evaluate_benchmark(
                    model, 
                    self._benchmarks[name], 
                    num_questions,
                    **config
                )
                for name in benchmark_names
            ]
            benchmark_results = await asyncio.gather(*tasks)
        else:
            # Run benchmarks sequentially
            benchmark_results = []
            for name in benchmark_names:
                result = await self._evaluate_benchmark(
                    model,
                    self._benchmarks[name],
                    num_questions,
                    **config
                )
                benchmark_results.append(result)
        
        # Add results to main results object
        for benchmark_result in benchmark_results:
            results.add_benchmark_result(benchmark_result)
        
        if save_results:
            self._results_history.append(results)
        
        logger.info(f"Evaluation completed. Overall score: {results.summary()['overall_score']:.3f}")
        return results
    
    async def _evaluate_benchmark(
        self,
        model: Model,
        benchmark: Benchmark,
        num_questions: Optional[int] = None,
        **config
    ) -> BenchmarkResult:
        """Evaluate model on a single benchmark."""
        logger.info(f"Evaluating {benchmark.name} benchmark")
        
        # Get questions
        questions = benchmark.get_questions()
        if num_questions:
            questions = questions[:num_questions]
        
        # Generate responses for all questions
        prompts = [q.prompt for q in questions]
        
        try:
            responses = await model.batch_generate(prompts, **config)
        except Exception as e:
            logger.error(f"Error generating responses for {benchmark.name}: {e}")
            # Fallback to sequential generation
            responses = []
            for prompt in prompts:
                try:
                    response = await model.generate(prompt, **config)
                    responses.append(response)
                except Exception as e:
                    logger.warning(f"Failed to generate response for question: {e}")
                    responses.append("")
        
        # Evaluate responses
        evaluation_results = []
        for question, response in zip(questions, responses):
            try:
                score = benchmark.evaluate_response(question, response)
                eval_result = EvaluationResult(
                    question_id=question.id,
                    question_prompt=question.prompt,
                    model_response=response,
                    score=score,
                    benchmark_name=benchmark.name,
                    category=question.category
                )
                evaluation_results.append(eval_result)
            except Exception as e:
                logger.error(f"Error evaluating question {question.id}: {e}")
                # Create a failed result
                from .benchmarks import Score
                failed_score = Score(value=0.0, passed=False, explanation=f"Evaluation error: {e}")
                eval_result = EvaluationResult(
                    question_id=question.id,
                    question_prompt=question.prompt,
                    model_response=response,
                    score=failed_score,
                    benchmark_name=benchmark.name,
                    category=question.category
                )
                evaluation_results.append(eval_result)
        
        benchmark_result = BenchmarkResult(
            benchmark_name=benchmark.name,
            model_name=model.name,
            model_provider=model.provider_name,
            results=evaluation_results,
            config=config
        )
        
        logger.info(f"Completed {benchmark.name}: {benchmark_result.average_score:.3f} avg score, {benchmark_result.pass_rate:.1f}% pass rate")
        return benchmark_result
    
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
        
        return benchmark