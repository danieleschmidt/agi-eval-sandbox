"""Model providers and abstractions."""

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


class RateLimits(BaseModel):
    """Rate limiting configuration."""
    requests_per_minute: int = 60
    tokens_per_minute: int = 100000
    max_concurrent: int = 10


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
        """Generate responses for multiple prompts."""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
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
        """Generate response using Anthropic API."""
        if httpx is None:
            raise ImportError("httpx is required for Anthropic provider. Install with: pip install httpx")
            
        payload = {
            "model": self.config.name,
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),
            "messages": [{"role": "user", "content": prompt}]
        }
        
        async with httpx.AsyncClient(timeout=self.config.timeout) as client:
            response = await client.post(
                f"{self.base_url}/messages",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            result = response.json()
            return result["content"][0]["text"]
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts."""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)
    
    def get_limits(self) -> RateLimits:
        """Get Anthropic rate limits."""
        return RateLimits(
            requests_per_minute=50,
            tokens_per_minute=40000,
            max_concurrent=10
        )


class LocalProvider(ModelProvider):
    """Local model provider for testing."""
    
    def __init__(self, config: ModelConfig):
        super().__init__(config)
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate mock response for testing."""
        return f"Mock response for: {prompt[:50]}..."
    
    async def batch_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate mock responses for testing."""
        return [await self.generate(prompt, **kwargs) for prompt in prompts]
    
    def get_limits(self) -> RateLimits:
        """Get unlimited rate limits for local testing."""
        return RateLimits(
            requests_per_minute=1000,
            tokens_per_minute=1000000,
            max_concurrent=100
        )


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
            raise ValueError(f"Unsupported provider: {provider}")
    
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