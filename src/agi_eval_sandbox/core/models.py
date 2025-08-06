"""Model providers and abstractions.

This module provides a unified interface for different AI model providers including OpenAI, 
Anthropic, and a mock provider for testing. All providers implement robust error handling,
rate limiting, and async batch processing.

Basic usage:

    # OpenAI model
    model = create_openai_model("gpt-3.5-turbo", api_key="sk-...")
    response = await model.generate("Hello world")
    
    # Anthropic model  
    model = create_anthropic_model("claude-3-haiku-20240307", api_key="sk-ant-...")
    response = await model.generate("Hello world")
    
    # Mock model for testing
    model = create_mock_model("test-model", simulate_delay=0.1, simulate_failures=True)
    response = await model.generate("Hello world")
    
    # Batch generation with concurrency control
    prompts = ["Hello", "How are you?", "Goodbye"]  
    responses = await model.batch_generate(prompts)

All models support:
- Robust error handling with retries and exponential backoff
- Rate limiting based on provider limits
- Structured logging with performance metrics
- Input validation and sanitization
- Async batch processing with configurable concurrency
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import asyncio
import json
import time

from .exceptions import ModelProviderError, ValidationError, TimeoutError, RateLimitError
from .validation import InputValidator
from .logging_config import get_logger, performance_logger

logger = get_logger("models")
try:
    from pydantic import BaseModel
except ImportError:
    # Fallback BaseModel implementation
    class BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

try:
    import httpx
except ImportError:
    httpx = None

try:
    import random
except ImportError:
    random = None


class RateLimits(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    max_concurrent: int = 10


class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, limits: RateLimits):
        self.limits = limits
        self.request_times = []
        self.token_usage = []
        self._lock = asyncio.Lock()
    
    async def acquire(self, estimated_tokens: int = 100) -> None:
        """Acquire permission to make a request."""
        async with self._lock:
            current_time = time.time()
            
            # Clean old entries (older than 1 minute)
            cutoff_time = current_time - 60
            self.request_times = [t for t in self.request_times if t > cutoff_time]
            self.token_usage = [(t, tokens) for t, tokens in self.token_usage if t > cutoff_time]
            
            # Check request rate limit
            if len(self.request_times) >= self.limits.requests_per_minute:
                wait_time = self.request_times[0] + 60 - current_time
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.2f}s", extra={
                        "requests_in_window": len(self.request_times),
                        "limit": self.limits.requests_per_minute,
                        "wait_time": wait_time
                    })
                    await asyncio.sleep(wait_time)
            
            # Check token rate limit
            current_tokens = sum(tokens for _, tokens in self.token_usage)
            if current_tokens + estimated_tokens > self.limits.tokens_per_minute:
                # Wait for oldest token usage to expire
                if self.token_usage:
                    wait_time = self.token_usage[0][0] + 60 - current_time
                    if wait_time > 0:
                        logger.info(f"Token rate limit reached, waiting {wait_time:.2f}s", extra={
                            "tokens_in_window": current_tokens,
                            "limit": self.limits.tokens_per_minute,
                            "wait_time": wait_time
                        })
                        await asyncio.sleep(wait_time)
            
            # Record this request
            self.request_times.append(current_time)
            self.token_usage.append((current_time, estimated_tokens))


@dataclass
class ModelConfig:
    """Model configuration with validation."""
    name: str
    provider: str
    api_key: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 2048
    timeout: int = 30
    retry_attempts: int = 3
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.name = InputValidator.validate_model_name(self.name)
        self.provider = InputValidator.validate_provider(self.provider)
        self.api_key = InputValidator.validate_api_key(self.api_key, self.provider)
        self.temperature = InputValidator.validate_temperature(self.temperature)
        self.max_tokens = InputValidator.validate_max_tokens(self.max_tokens)
        
        if not isinstance(self.timeout, int) or self.timeout <= 0:
            raise ValidationError("Timeout must be a positive integer")
        
        if not isinstance(self.retry_attempts, int) or self.retry_attempts < 0:
            raise ValidationError("Retry attempts must be a non-negative integer")


class ModelProvider(ABC):
    """Abstract base class for model providers."""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate a response from the model."""
        pass
    
    @abstractmethod  
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        pass
    
    @abstractmethod
    def get_limits(self) -> RateLimits:
        """Get rate limiting information."""
        pass


class OpenAIProvider(ModelProvider):
    """OpenAI model provider."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("OpenAI API key is required")
        self.base_url = "https://api.openai.com/v1"
        self.headers = {
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json"
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using OpenAI API with robust error handling."""
        if httpx is None:
            raise ModelProviderError("httpx is required for OpenAI provider. Install with: pip install httpx")
        
        # Validate and sanitize input
        prompt = InputValidator.sanitize_user_input(prompt)
        if not prompt.strip():
            raise ValidationError("Prompt cannot be empty")
        
        start_time = time.time()
        attempt = 0
        
        while attempt < self.config.retry_attempts:
            try:
                payload = {
                    "model": self.config.name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": kwargs.get("temperature", self.config.temperature),
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens)
                }
                
                logger.debug(f"OpenAI API request attempt {attempt + 1}", extra={
                    "model": self.config.name,
                    "prompt_length": len(prompt),
                    "temperature": payload["temperature"],
                    "max_tokens": payload["max_tokens"]
                })
                
                async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=self.headers,
                        json=payload
                    )
                    
                    if response.status_code == 429:  # Rate limit
                        attempt += 1
                        if attempt < self.config.retry_attempts:
                            wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry", extra={
                                "attempt": attempt,
                                "wait_time": wait_time
                            })
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise RateLimitError("OpenAI API rate limit exceeded")
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if "choices" not in result or not result["choices"]:
                        raise ModelProviderError("Invalid response format from OpenAI API")
                    
                    generated_text = result["choices"][0]["message"]["content"]
                    duration_seconds = time.time() - start_time
                    
                    # Log performance metrics
                    performance_logger.log_api_performance(
                        endpoint="openai_chat_completions",
                        method="POST",
                        duration_ms=duration_seconds * 1000,
                        status_code=response.status_code,
                        response_size_bytes=len(response.content)
                    )
                    
                    logger.info("OpenAI API request successful", extra={
                        "model": self.config.name,
                        "duration_seconds": duration_seconds,
                        "response_length": len(generated_text),
                        "attempt": attempt + 1
                    })
                    
                    return generated_text
                    
            except httpx.TimeoutException:
                attempt += 1
                if attempt < self.config.retry_attempts:
                    logger.warning(f"OpenAI API timeout, retrying (attempt {attempt + 1})")
                    await asyncio.sleep(1)
                    continue
                else:
                    raise TimeoutError(f"OpenAI API request timed out after {self.config.retry_attempts} attempts")
            
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:  # Server errors, retry
                    attempt += 1
                    if attempt < self.config.retry_attempts:
                        logger.warning(f"OpenAI API server error {e.response.status_code}, retrying")
                        await asyncio.sleep(2 ** attempt)
                        continue
                
                # Client errors, don't retry
                error_detail = ""
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("error", {}).get("message", "")
                except:
                    error_detail = e.response.text
                
                raise ModelProviderError(f"OpenAI API error {e.response.status_code}: {error_detail}")
            
            except Exception as e:
                attempt += 1
                if attempt < self.config.retry_attempts:
                    logger.warning(f"Unexpected error in OpenAI API call, retrying: {str(e)}")
                    await asyncio.sleep(1)
                    continue
                else:
                    raise ModelProviderError(f"OpenAI API request failed: {str(e)}")
        
        raise ModelProviderError(f"OpenAI API request failed after {self.config.retry_attempts} attempts")
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts with concurrency control."""
        if not prompts:
            return []
        
        # Get rate limits for concurrency control
        limits = self.get_limits()
        semaphore = asyncio.Semaphore(limits.max_concurrent)
        
        async def generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.generate(prompt, **kwargs)
        
        logger.info(f"Starting batch generation for {len(prompts)} prompts", extra={
            "model": self.config.name,
            "batch_size": len(prompts),
            "max_concurrent": limits.max_concurrent
        })
        
        start_time = time.time()
        try:
            tasks = [generate_with_semaphore(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            final_results = []
            failed_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_count += 1
                    logger.error(f"Failed to generate response for prompt {i}", extra={
                        "prompt_index": i,
                        "error": str(result),
                        "prompt_preview": prompts[i][:100] + "..." if len(prompts[i]) > 100 else prompts[i]
                    })
                    # Return error placeholder instead of raising
                    final_results.append(f"[ERROR: {str(result)}]")
                else:
                    final_results.append(result)
            
            duration_seconds = time.time() - start_time
            logger.info(f"Batch generation completed", extra={
                "model": self.config.name,
                "batch_size": len(prompts),
                "successful": len(prompts) - failed_count,
                "failed": failed_count,
                "duration_seconds": duration_seconds,
                "prompts_per_second": len(prompts) / duration_seconds if duration_seconds > 0 else 0
            })
            
            return final_results
            
        except Exception as e:
            logger.error(f"Batch generation failed", extra={
                "model": self.config.name,
                "batch_size": len(prompts),
                "error": str(e)
            })
            raise ModelProviderError(f"Batch generation failed: {str(e)}")
    
    def get_limits(self) -> RateLimits:
        """Get OpenAI rate limits."""
        return RateLimits(
            requests_per_minute=60,
            tokens_per_minute=90000,
            max_concurrent=20
        )


class AnthropicProvider(ModelProvider):
    """Anthropic model provider."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        if not config.api_key:
            raise ValueError("Anthropic API key is required")
        self.base_url = "https://api.anthropic.com/v1"  
        self.headers = {
            "x-api-key": config.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using Anthropic API with robust error handling."""
        if httpx is None:
            raise ModelProviderError("httpx is required for Anthropic provider. Install with: pip install httpx")
        
        # Validate and sanitize input
        prompt = InputValidator.sanitize_user_input(prompt)
        if not prompt.strip():
            raise ValidationError("Prompt cannot be empty")
        
        start_time = time.time()
        attempt = 0
        
        while attempt < self.config.retry_attempts:
            try:
                payload = {
                    "model": self.config.name,
                    "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
                    "messages": [{"role": "user", "content": prompt}]
                }
                
                # Add temperature if specified (Anthropic uses different parameter name)
                temperature = kwargs.get("temperature", self.config.temperature)
                if temperature > 0:
                    payload["temperature"] = temperature
                
                logger.debug(f"Anthropic API request attempt {attempt + 1}", extra={
                    "model": self.config.name,
                    "prompt_length": len(prompt),
                    "temperature": temperature,
                    "max_tokens": payload["max_tokens"]
                })
                
                async with httpx.AsyncClient(timeout=self.config.timeout) as client:
                    response = await client.post(
                        f"{self.base_url}/messages",
                        headers=self.headers,
                        json=payload
                    )
                    
                    if response.status_code == 429:  # Rate limit
                        attempt += 1
                        if attempt < self.config.retry_attempts:
                            wait_time = min(2 ** attempt, 60)  # Exponential backoff, max 60s
                            logger.warning(f"Rate limited, waiting {wait_time}s before retry", extra={
                                "attempt": attempt,
                                "wait_time": wait_time
                            })
                            await asyncio.sleep(wait_time)
                            continue
                        else:
                            raise RateLimitError("Anthropic API rate limit exceeded")
                    
                    response.raise_for_status()
                    result = response.json()
                    
                    if "content" not in result or not result["content"]:
                        raise ModelProviderError("Invalid response format from Anthropic API")
                    
                    generated_text = result["content"][0]["text"]
                    duration_seconds = time.time() - start_time
                    
                    # Log performance metrics
                    performance_logger.log_api_performance(
                        endpoint="anthropic_messages",
                        method="POST",
                        duration_ms=duration_seconds * 1000,
                        status_code=response.status_code,
                        response_size_bytes=len(response.content)
                    )
                    
                    logger.info("Anthropic API request successful", extra={
                        "model": self.config.name,
                        "duration_seconds": duration_seconds,
                        "response_length": len(generated_text),
                        "attempt": attempt + 1
                    })
                    
                    return generated_text
                    
            except httpx.TimeoutException:
                attempt += 1
                if attempt < self.config.retry_attempts:
                    logger.warning(f"Anthropic API timeout, retrying (attempt {attempt + 1})")
                    await asyncio.sleep(1)
                    continue
                else:
                    raise TimeoutError(f"Anthropic API request timed out after {self.config.retry_attempts} attempts")
            
            except httpx.HTTPStatusError as e:
                if e.response.status_code >= 500:  # Server errors, retry
                    attempt += 1
                    if attempt < self.config.retry_attempts:
                        logger.warning(f"Anthropic API server error {e.response.status_code}, retrying")
                        await asyncio.sleep(2 ** attempt)
                        continue
                
                # Client errors, don't retry
                error_detail = ""
                try:
                    error_data = e.response.json()
                    error_detail = error_data.get("error", {}).get("message", "")
                except:
                    error_detail = e.response.text
                
                raise ModelProviderError(f"Anthropic API error {e.response.status_code}: {error_detail}")
            
            except Exception as e:
                attempt += 1
                if attempt < self.config.retry_attempts:
                    logger.warning(f"Unexpected error in Anthropic API call, retrying: {str(e)}")
                    await asyncio.sleep(1)
                    continue
                else:
                    raise ModelProviderError(f"Anthropic API request failed: {str(e)}")
        
        raise ModelProviderError(f"Anthropic API request failed after {self.config.retry_attempts} attempts")
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts with concurrency control."""
        if not prompts:
            return []
        
        # Get rate limits for concurrency control
        limits = self.get_limits()
        semaphore = asyncio.Semaphore(limits.max_concurrent)
        
        async def generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.generate(prompt, **kwargs)
        
        logger.info(f"Starting batch generation for {len(prompts)} prompts", extra={
            "model": self.config.name,
            "batch_size": len(prompts),
            "max_concurrent": limits.max_concurrent
        })
        
        start_time = time.time()
        try:
            tasks = [generate_with_semaphore(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            final_results = []
            failed_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_count += 1
                    logger.error(f"Failed to generate response for prompt {i}", extra={
                        "prompt_index": i,
                        "error": str(result),
                        "prompt_preview": prompts[i][:100] + "..." if len(prompts[i]) > 100 else prompts[i]
                    })
                    # Return error placeholder instead of raising
                    final_results.append(f"[ERROR: {str(result)}]")
                else:
                    final_results.append(result)
            
            duration_seconds = time.time() - start_time
            logger.info(f"Batch generation completed", extra={
                "model": self.config.name,
                "batch_size": len(prompts),
                "successful": len(prompts) - failed_count,
                "failed": failed_count,
                "duration_seconds": duration_seconds,
                "prompts_per_second": len(prompts) / duration_seconds if duration_seconds > 0 else 0
            })
            
            return final_results
            
        except Exception as e:
            logger.error(f"Batch generation failed", extra={
                "model": self.config.name,
                "batch_size": len(prompts),
                "error": str(e)
            })
            raise ModelProviderError(f"Batch generation failed: {str(e)}")
    
    def get_limits(self) -> RateLimits:
        """Get Anthropic rate limits."""
        return RateLimits(
            requests_per_minute=50,
            tokens_per_minute=40000,
            max_concurrent=10
        )


class LocalProvider(ModelProvider):
    """Local model provider for testing with configurable behavior."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Test configuration options
        self.simulate_delay = getattr(config, 'simulate_delay', 0.1)  # Default 100ms delay
        self.simulate_failures = getattr(config, 'simulate_failures', False)
        self.failure_rate = getattr(config, 'failure_rate', 0.1)  # 10% failure rate
        self.response_template = getattr(config, 'response_template', None)
        self._call_count = 0
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response for testing with configurable behavior."""
        self._call_count += 1
        
        # Validate and sanitize input like real providers
        prompt = InputValidator.sanitize_user_input(prompt)
        if not prompt.strip():
            raise ValidationError("Prompt cannot be empty")
        
        start_time = time.time()
        
        # Simulate network delay
        if self.simulate_delay > 0:
            await asyncio.sleep(self.simulate_delay)
        
        # Simulate random failures for testing error handling
        if self.simulate_failures and random:
            if random.random() < self.failure_rate:
                error_types = [
                    RateLimitError("Mock rate limit exceeded"),
                    TimeoutError("Mock timeout"),
                    ModelProviderError("Mock provider error")
                ]
                raise random.choice(error_types)
        
        # Generate response based on template or default
        if self.response_template:
            response = self.response_template.format(
                prompt=prompt[:100],
                model=self.config.name,
                call_count=self._call_count,
                temperature=kwargs.get("temperature", self.config.temperature),
                max_tokens=kwargs.get("max_tokens", self.config.max_tokens)
            )
        else:
            response = f"Mock response from {self.config.name} (call #{self._call_count}) for: {prompt[:50]}..."
        
        duration_seconds = time.time() - start_time
        
        # Log like real providers
        logger.info("Local provider request completed", extra={
            "model": self.config.name,
            "duration_seconds": duration_seconds,
            "response_length": len(response),
            "call_count": self._call_count,
            "simulated_delay": self.simulate_delay
        })
        
        return response
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate mock responses for testing with concurrency control."""
        if not prompts:
            return []
        
        # Get rate limits for concurrency control (even for mock)
        limits = self.get_limits()
        semaphore = asyncio.Semaphore(limits.max_concurrent)
        
        async def generate_with_semaphore(prompt: str) -> str:
            async with semaphore:
                return await self.generate(prompt, **kwargs)
        
        logger.info(f"Starting mock batch generation for {len(prompts)} prompts", extra={
            "model": self.config.name,
            "batch_size": len(prompts),
            "max_concurrent": limits.max_concurrent,
            "simulate_delay": self.simulate_delay,
            "simulate_failures": self.simulate_failures
        })
        
        start_time = time.time()
        try:
            tasks = [generate_with_semaphore(prompt) for prompt in prompts]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results and handle exceptions
            final_results = []
            failed_count = 0
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    failed_count += 1
                    logger.warning(f"Mock generation failed for prompt {i}", extra={
                        "prompt_index": i,
                        "error": str(result),
                        "prompt_preview": prompts[i][:100] + "..." if len(prompts[i]) > 100 else prompts[i]
                    })
                    # Return error placeholder instead of raising
                    final_results.append(f"[MOCK ERROR: {str(result)}]")
                else:
                    final_results.append(result)
            
            duration_seconds = time.time() - start_time
            logger.info(f"Mock batch generation completed", extra={
                "model": self.config.name,
                "batch_size": len(prompts),
                "successful": len(prompts) - failed_count,
                "failed": failed_count,
                "duration_seconds": duration_seconds,
                "prompts_per_second": len(prompts) / duration_seconds if duration_seconds > 0 else 0
            })
            
            return final_results
            
        except Exception as e:
            logger.error(f"Mock batch generation failed", extra={
                "model": self.config.name,
                "batch_size": len(prompts),
                "error": str(e)
            })
            raise ModelProviderError(f"Mock batch generation failed: {str(e)}")
    
    def get_limits(self) -> RateLimits:
        """Get mock rate limits for testing."""
        return RateLimits(
            requests_per_minute=1000,
            tokens_per_minute=1000000,
            max_concurrent=50  # Slightly lower to test concurrency control
        )
    
    def reset_call_count(self) -> None:
        """Reset call count for testing."""
        self._call_count = 0
    
    def get_call_count(self) -> int:
        """Get current call count for testing."""
        return self._call_count


class Model:
    """High-level model interface."""
    
    def __init__(
        self,
        provider: str,
        name: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize model with provider and configuration."""
        config = ModelConfig(
            name=name,
            provider=provider,
            api_key=api_key,
            **kwargs
        )
        
        # Create provider instance
        if provider.lower() == "openai":
            self.provider = OpenAIProvider(config)
        elif provider.lower() == "anthropic":
            self.provider = AnthropicProvider(config)
        elif provider.lower() == "local":
            self.provider = LocalProvider(config)
        else:
            raise ValueError(f"Unsupported provider: {provider}. Supported providers: openai, anthropic, local")
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate response from model."""
        return await self.provider.generate(prompt, **kwargs)
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        return await self.provider.batch_generate(prompts, **kwargs)
    
    def get_limits(self) -> RateLimits:
        """Get rate limiting information."""
        return self.provider.get_limits()
    
    @property
    def name(self) -> str:
        """Get model name."""
        return self.provider.config.name
    
    @property
    def provider_name(self) -> str:
        """Get provider name."""
        return self.provider.config.provider


# Utility functions for creating models

def create_openai_model(
    model_name: str = "gpt-3.5-turbo",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **kwargs
) -> Model:
    """Create an OpenAI model with sensible defaults."""
    return Model(
        provider="openai",
        name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def create_anthropic_model(
    model_name: str = "claude-3-haiku-20240307",
    api_key: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    **kwargs
) -> Model:
    """Create an Anthropic model with sensible defaults."""
    return Model(
        provider="anthropic",
        name=model_name,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs
    )


def create_mock_model(
    model_name: str = "mock-model",
    simulate_delay: float = 0.1,
    simulate_failures: bool = False,
    failure_rate: float = 0.1,
    response_template: Optional[str] = None,
    **kwargs
) -> Model:
    """Create a mock model for testing."""
    # Create a custom config class that includes mock-specific attributes
    class MockModelConfig(ModelConfig):
        def __init__(self, *args, **kwargs):
            # Extract mock-specific parameters
            self.simulate_delay = kwargs.pop('simulate_delay', 0.1)
            self.simulate_failures = kwargs.pop('simulate_failures', False)
            self.failure_rate = kwargs.pop('failure_rate', 0.1)
            self.response_template = kwargs.pop('response_template', None)
            super().__init__(*args, **kwargs)
    
    # Create the mock config
    mock_config = MockModelConfig(
        provider="local",
        name=model_name,
        simulate_delay=simulate_delay,
        simulate_failures=simulate_failures,
        failure_rate=failure_rate,
        response_template=response_template,
        **kwargs
    )
    
    # Create a mock model by directly instantiating the provider
    mock_provider = LocalProvider(mock_config)
    
    # Create a model wrapper
    class MockModel(Model):
        def __init__(self, provider):
            self.provider = provider
    
    return MockModel(mock_provider)


def get_available_providers() -> List[str]:
    """Get list of available model providers."""
    return ["openai", "anthropic", "local"]


def validate_model_config(provider: str, model_name: str, api_key: Optional[str] = None) -> bool:
    """Validate that a model configuration is valid."""
    try:
        # Test basic configuration
        config = ModelConfig(
            provider=provider,
            name=model_name,
            api_key=api_key
        )
        
        # Test provider creation
        if provider.lower() == "openai":
            OpenAIProvider(config)
        elif provider.lower() == "anthropic":
            AnthropicProvider(config)
        elif provider.lower() == "local":
            LocalProvider(config)
        else:
            return False
            
        return True
    except Exception:
        return False