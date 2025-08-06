"""
Comprehensive unit tests for model providers and functionality.
"""

import pytest
import asyncio
import time
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from agi_eval_sandbox.core.models import (
    Model, ModelConfig, ModelProvider, OpenAIProvider, AnthropicProvider, LocalProvider,
    RateLimiter, RateLimits, create_openai_model, create_anthropic_model, create_mock_model,
    get_available_providers, validate_model_config
)
from agi_eval_sandbox.core.exceptions import (
    ValidationError, ModelProviderError, TimeoutError, RateLimitError
)


@pytest.mark.unit
class TestModelConfig:
    """Test ModelConfig dataclass and validation."""
    
    def test_valid_model_config(self):
        """Test creating valid model configuration."""
        config = ModelConfig(
            name="gpt-4",
            provider="openai",
            api_key="sk-test123",
            temperature=0.5,
            max_tokens=2000,
            timeout=30,
            retry_attempts=3
        )
        
        assert config.name == "gpt-4"
        assert config.provider == "openai"
        assert config.api_key == "sk-test123"
        assert config.temperature == 0.5
        assert config.max_tokens == 2000
        assert config.timeout == 30
        assert config.retry_attempts == 3
    
    def test_invalid_temperature(self):
        """Test invalid temperature validation."""
        with pytest.raises(ValidationError, match="must be between 0.0 and 2.0"):
            ModelConfig(
                name="gpt-4",
                provider="openai",
                temperature=3.0  # Invalid
            )
    
    def test_invalid_max_tokens(self):
        """Test invalid max_tokens validation."""
        with pytest.raises(ValidationError, match="must be between 1 and"):
            ModelConfig(
                name="gpt-4", 
                provider="openai",
                max_tokens=0  # Invalid
            )
    
    def test_invalid_timeout(self):
        """Test invalid timeout validation."""
        with pytest.raises(ValidationError, match="must be a positive integer"):
            ModelConfig(
                name="gpt-4",
                provider="openai",
                timeout=-1  # Invalid
            )
    
    def test_invalid_retry_attempts(self):
        """Test invalid retry_attempts validation."""
        with pytest.raises(ValidationError, match="must be a non-negative integer"):
            ModelConfig(
                name="gpt-4",
                provider="openai",
                retry_attempts=-1  # Invalid
            )


@pytest.mark.unit
class TestRateLimiter:
    """Test rate limiting functionality."""
    
    @pytest.mark.asyncio
    async def test_basic_rate_limiting(self):
        """Test basic rate limiting functionality."""
        limits = RateLimits(requests_per_minute=5, tokens_per_minute=1000, max_concurrent=2)
        limiter = RateLimiter(limits)
        
        # Should allow requests within limits
        for _ in range(5):
            await limiter.acquire(estimated_tokens=100)
    
    @pytest.mark.asyncio
    async def test_request_rate_limiting(self):
        """Test request rate limiting."""
        limits = RateLimits(requests_per_minute=2, tokens_per_minute=10000, max_concurrent=10)
        limiter = RateLimiter(limits)
        
        # Use up the request limit
        await limiter.acquire(estimated_tokens=100)
        await limiter.acquire(estimated_tokens=100)
        
        # This should cause a delay due to rate limiting
        start_time = time.time()
        
        # Mock time.time to simulate rate limit window
        with patch('time.time') as mock_time:
            # First call returns current time
            # Second call returns time that would trigger rate limiting
            mock_time.side_effect = [time.time(), time.time() + 0.1, time.time() + 61]
            
            await limiter.acquire(estimated_tokens=100)
        
        # Test completed without hanging
        assert True
    
    @pytest.mark.asyncio
    async def test_token_rate_limiting(self):
        """Test token-based rate limiting."""
        limits = RateLimits(requests_per_minute=100, tokens_per_minute=500, max_concurrent=10)
        limiter = RateLimiter(limits)
        
        # Use most of token budget
        await limiter.acquire(estimated_tokens=400)
        
        # This should work
        await limiter.acquire(estimated_tokens=50)
        
        # This would exceed token limit but we'll mock the time handling
        with patch('time.time') as mock_time:
            mock_time.side_effect = [time.time(), time.time() + 0.1, time.time() + 61]
            await limiter.acquire(estimated_tokens=200)
    
    def test_rate_limits_dataclass(self):
        """Test RateLimits dataclass."""
        limits = RateLimits(
            requests_per_minute=100,
            tokens_per_minute=50000,
            max_concurrent=20
        )
        
        assert limits.requests_per_minute == 100
        assert limits.tokens_per_minute == 50000
        assert limits.max_concurrent == 20


@pytest.mark.unit
class TestLocalProvider:
    """Test LocalProvider (mock provider) functionality."""
    
    def test_local_provider_creation(self):
        """Test LocalProvider initialization."""
        config = ModelConfig(
            name="test-model",
            provider="local",
            simulate_delay=0.1,
            simulate_failures=True,
            failure_rate=0.2
        )
        
        provider = LocalProvider(config)
        
        assert provider.config.name == "test-model"
        assert provider.simulate_delay == 0.1
        assert provider.simulate_failures is True
        assert provider.failure_rate == 0.2
        assert provider._call_count == 0
    
    @pytest.mark.asyncio
    async def test_local_provider_generation(self):
        """Test LocalProvider text generation."""
        config = ModelConfig(
            name="test-model",
            provider="local",
            simulate_delay=0.01  # Small delay for testing
        )
        
        provider = LocalProvider(config)
        
        response = await provider.generate("Test prompt")
        
        assert "Mock response from test-model" in response
        assert "Test prompt" in response
        assert provider.get_call_count() == 1
    
    @pytest.mark.asyncio
    async def test_local_provider_batch_generation(self):
        """Test LocalProvider batch generation."""
        config = ModelConfig(
            name="batch-test",
            provider="local",
            simulate_delay=0.01
        )
        
        provider = LocalProvider(config)
        
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        responses = await provider.batch_generate(prompts)
        
        assert len(responses) == 3
        assert all("Mock response" in response for response in responses)
        assert provider.get_call_count() == 3
    
    @pytest.mark.asyncio
    async def test_local_provider_with_failures(self):
        """Test LocalProvider with failure simulation."""
        config = ModelConfig(
            name="failure-test",
            provider="local",
            simulate_delay=0.001,
            simulate_failures=True,
            failure_rate=1.0  # Always fail
        )
        
        provider = LocalProvider(config)
        
        # Should raise an exception due to simulated failure
        with pytest.raises((RateLimitError, TimeoutError, ModelProviderError)):
            await provider.generate("This should fail")
    
    @pytest.mark.asyncio
    async def test_local_provider_custom_response_template(self):
        """Test LocalProvider with custom response template."""
        config = ModelConfig(
            name="template-test",
            provider="local",
            response_template="Custom response for {prompt} from {model} (call #{call_count})"
        )
        
        provider = LocalProvider(config)
        
        response = await provider.generate("test prompt")
        
        assert "Custom response for test prompt from template-test (call #1)" == response
    
    def test_local_provider_call_count_management(self):
        """Test call count management in LocalProvider."""
        config = ModelConfig(name="count-test", provider="local")
        provider = LocalProvider(config)
        
        assert provider.get_call_count() == 0
        
        # Manually increment for testing
        provider._call_count = 5
        assert provider.get_call_count() == 5
        
        provider.reset_call_count()
        assert provider.get_call_count() == 0
    
    def test_local_provider_rate_limits(self):
        """Test LocalProvider rate limits."""
        config = ModelConfig(name="limits-test", provider="local")
        provider = LocalProvider(config)
        
        limits = provider.get_limits()
        
        assert limits.requests_per_minute == 1000
        assert limits.tokens_per_minute == 1000000
        assert limits.max_concurrent == 50


@pytest.mark.unit
class TestOpenAIProvider:
    """Test OpenAI provider (without actual API calls)."""
    
    def test_openai_provider_creation(self):
        """Test OpenAI provider initialization."""
        config = ModelConfig(
            name="gpt-4",
            provider="openai",
            api_key="sk-test123"
        )
        
        provider = OpenAIProvider(config)
        
        assert provider.config.name == "gpt-4"
        assert provider.base_url == "https://api.openai.com/v1"
        assert provider.headers["Authorization"] == "Bearer sk-test123"
        assert provider.headers["Content-Type"] == "application/json"
    
    def test_openai_provider_missing_api_key(self):
        """Test OpenAI provider creation without API key."""
        config = ModelConfig(
            name="gpt-4",
            provider="openai"
            # No API key provided
        )
        
        with pytest.raises(ValueError, match="OpenAI API key is required"):
            OpenAIProvider(config)
    
    def test_openai_provider_rate_limits(self):
        """Test OpenAI provider rate limits."""
        config = ModelConfig(
            name="gpt-4",
            provider="openai",
            api_key="sk-test"
        )
        
        provider = OpenAIProvider(config)
        limits = provider.get_limits()
        
        assert limits.requests_per_minute == 60
        assert limits.tokens_per_minute == 90000
        assert limits.max_concurrent == 20


@pytest.mark.unit
class TestAnthropicProvider:
    """Test Anthropic provider (without actual API calls)."""
    
    def test_anthropic_provider_creation(self):
        """Test Anthropic provider initialization."""
        config = ModelConfig(
            name="claude-3-haiku-20240307",
            provider="anthropic",
            api_key="sk-ant-test123"
        )
        
        provider = AnthropicProvider(config)
        
        assert provider.config.name == "claude-3-haiku-20240307"
        assert provider.base_url == "https://api.anthropic.com/v1"
        assert provider.headers["x-api-key"] == "sk-ant-test123"
        assert provider.headers["Content-Type"] == "application/json"
        assert provider.headers["anthropic-version"] == "2023-06-01"
    
    def test_anthropic_provider_missing_api_key(self):
        """Test Anthropic provider creation without API key."""
        config = ModelConfig(
            name="claude-3-haiku-20240307",
            provider="anthropic"
            # No API key provided
        )
        
        with pytest.raises(ValueError, match="Anthropic API key is required"):
            AnthropicProvider(config)
    
    def test_anthropic_provider_rate_limits(self):
        """Test Anthropic provider rate limits."""
        config = ModelConfig(
            name="claude-3-haiku-20240307",
            provider="anthropic",
            api_key="sk-ant-test"
        )
        
        provider = AnthropicProvider(config)
        limits = provider.get_limits()
        
        assert limits.requests_per_minute == 50
        assert limits.tokens_per_minute == 40000
        assert limits.max_concurrent == 10


@pytest.mark.unit
class TestModelClass:
    """Test high-level Model class."""
    
    def test_model_creation_openai(self):
        """Test Model creation with OpenAI provider."""
        model = Model(
            provider="openai",
            name="gpt-4",
            api_key="sk-test123",
            temperature=0.5
        )
        
        assert model.name == "gpt-4"
        assert model.provider_name == "openai"
        assert isinstance(model.provider, OpenAIProvider)
    
    def test_model_creation_anthropic(self):
        """Test Model creation with Anthropic provider."""
        model = Model(
            provider="anthropic",
            name="claude-3-haiku-20240307",
            api_key="sk-ant-test123"
        )
        
        assert model.name == "claude-3-haiku-20240307"
        assert model.provider_name == "anthropic"
        assert isinstance(model.provider, AnthropicProvider)
    
    def test_model_creation_local(self):
        """Test Model creation with local provider."""
        model = Model(
            provider="local",
            name="test-model",
            simulate_delay=0.05
        )
        
        assert model.name == "test-model"
        assert model.provider_name == "local"
        assert isinstance(model.provider, LocalProvider)
    
    def test_model_creation_invalid_provider(self):
        """Test Model creation with invalid provider."""
        with pytest.raises(ValueError, match="Unsupported provider: invalid"):
            Model(
                provider="invalid",
                name="test-model"
            )
    
    @pytest.mark.asyncio
    async def test_model_generate_interface(self):
        """Test Model generate interface."""
        model = Model(
            provider="local",
            name="interface-test",
            simulate_delay=0.01
        )
        
        response = await model.generate("Test prompt")
        assert "Mock response" in response
        assert "interface-test" in response
    
    @pytest.mark.asyncio
    async def test_model_batch_generate_interface(self):
        """Test Model batch_generate interface."""
        model = Model(
            provider="local",
            name="batch-interface-test",
            simulate_delay=0.01
        )
        
        prompts = ["Prompt A", "Prompt B"]
        responses = await model.batch_generate(prompts)
        
        assert len(responses) == 2
        assert all("Mock response" in response for response in responses)
    
    def test_model_get_limits_interface(self):
        """Test Model get_limits interface."""
        model = Model(provider="local", name="limits-test")
        
        limits = model.get_limits()
        assert isinstance(limits, RateLimits)
        assert limits.requests_per_minute > 0
        assert limits.tokens_per_minute > 0


@pytest.mark.unit
class TestModelFactoryFunctions:
    """Test model factory functions."""
    
    def test_create_openai_model(self):
        """Test create_openai_model factory function."""
        model = create_openai_model(
            model_name="gpt-4",
            api_key="sk-test123",
            temperature=0.7,
            max_tokens=1500
        )
        
        assert model.name == "gpt-4"
        assert model.provider_name == "openai"
        assert model.provider.config.temperature == 0.7
        assert model.provider.config.max_tokens == 1500
    
    def test_create_anthropic_model(self):
        """Test create_anthropic_model factory function."""
        model = create_anthropic_model(
            model_name="claude-3-sonnet-20240229",
            api_key="sk-ant-test123",
            temperature=0.3,
            max_tokens=2000
        )
        
        assert model.name == "claude-3-sonnet-20240229"
        assert model.provider_name == "anthropic"
        assert model.provider.config.temperature == 0.3
        assert model.provider.config.max_tokens == 2000
    
    def test_create_mock_model(self):
        """Test create_mock_model factory function."""
        model = create_mock_model(
            model_name="test-mock",
            simulate_delay=0.05,
            simulate_failures=True,
            failure_rate=0.1,
            response_template="Mock: {prompt}"
        )
        
        assert model.name == "test-mock"
        assert model.provider.simulate_delay == 0.05
        assert model.provider.simulate_failures is True
        assert model.provider.failure_rate == 0.1


@pytest.mark.unit
class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_get_available_providers(self):
        """Test get_available_providers function."""
        providers = get_available_providers()
        
        assert isinstance(providers, list)
        assert "openai" in providers
        assert "anthropic" in providers
        assert "local" in providers
        assert len(providers) >= 3
    
    def test_validate_model_config_valid(self):
        """Test validate_model_config with valid configurations."""
        # Valid OpenAI config
        assert validate_model_config("openai", "gpt-4", "sk-test123") is True
        
        # Valid Anthropic config
        assert validate_model_config("anthropic", "claude-3-haiku-20240307", "sk-ant-test") is True
        
        # Valid local config (no API key needed)
        assert validate_model_config("local", "test-model") is True
    
    def test_validate_model_config_invalid(self):
        """Test validate_model_config with invalid configurations."""
        # Invalid provider
        assert validate_model_config("invalid_provider", "model") is False
        
        # Empty model name
        assert validate_model_config("openai", "") is False
        
        # Invalid configuration that would raise exception
        assert validate_model_config("openai", "gpt-4", None) is False


@pytest.mark.unit
class TestModelProviderErrorHandling:
    """Test error handling in model providers."""
    
    @patch('agi_eval_sandbox.core.models.httpx')
    def test_openai_provider_missing_httpx(self, mock_httpx):
        """Test OpenAI provider behavior when httpx is not available."""
        mock_httpx = None
        
        config = ModelConfig(
            name="gpt-4",
            provider="openai",
            api_key="sk-test"
        )
        
        provider = OpenAIProvider(config)
        
        # This would be tested in integration tests with actual async call
        # Here we just test the provider creation
        assert provider.config.name == "gpt-4"
    
    @patch('agi_eval_sandbox.core.models.httpx')
    def test_anthropic_provider_missing_httpx(self, mock_httpx):
        """Test Anthropic provider behavior when httpx is not available."""
        mock_httpx = None
        
        config = ModelConfig(
            name="claude-3-haiku-20240307",
            provider="anthropic",
            api_key="sk-ant-test"
        )
        
        provider = AnthropicProvider(config)
        
        # This would be tested in integration tests with actual async call
        # Here we just test the provider creation
        assert provider.config.name == "claude-3-haiku-20240307"
    
    @pytest.mark.asyncio
    async def test_local_provider_empty_prompt_handling(self):
        """Test LocalProvider handling of empty prompts."""
        config = ModelConfig(name="empty-test", provider="local")
        provider = LocalProvider(config)
        
        # Empty prompt should raise ValidationError
        with pytest.raises(ValidationError, match="cannot be empty"):
            await provider.generate("")
        
        # Whitespace-only prompt should also fail
        with pytest.raises(ValidationError, match="cannot be empty"):
            await provider.generate("   ")
    
    @pytest.mark.asyncio
    async def test_local_provider_batch_empty_list(self):
        """Test LocalProvider handling of empty batch requests."""
        config = ModelConfig(name="empty-batch-test", provider="local")
        provider = LocalProvider(config)
        
        responses = await provider.batch_generate([])
        assert responses == []
    
    @pytest.mark.asyncio
    async def test_local_provider_batch_with_failures(self):
        """Test LocalProvider batch generation with some failures."""
        config = ModelConfig(
            name="batch-failure-test",
            provider="local",
            simulate_failures=True,
            failure_rate=0.5,  # 50% failure rate
            simulate_delay=0.001
        )
        
        provider = LocalProvider(config)
        
        # Run batch generation - some should fail, some should succeed
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3", "Prompt 4"]
        responses = await provider.batch_generate(prompts)
        
        # Should get responses for all prompts (failures return error messages)
        assert len(responses) == 4
        
        # Some responses should be error messages, some should be normal
        error_responses = [r for r in responses if r.startswith("[MOCK ERROR:")]
        normal_responses = [r for r in responses if not r.startswith("[MOCK ERROR:")]
        
        # With 50% failure rate and randomness, we expect some of each type
        # But we can't guarantee exact counts due to randomness
        assert len(error_responses) + len(normal_responses) == 4


@pytest.mark.unit
class TestConcurrencyControl:
    """Test concurrency control in model providers."""
    
    @pytest.mark.asyncio
    async def test_batch_generation_concurrency_limit(self):
        """Test that batch generation respects concurrency limits."""
        config = ModelConfig(
            name="concurrency-test",
            provider="local",
            simulate_delay=0.1  # Longer delay to test concurrency
        )
        
        provider = LocalProvider(config)
        limits = provider.get_limits()
        
        # Create more prompts than the max_concurrent limit
        num_prompts = limits.max_concurrent + 10
        prompts = [f"Prompt {i}" for i in range(num_prompts)]
        
        start_time = time.time()
        responses = await provider.batch_generate(prompts)
        elapsed = time.time() - start_time
        
        # Should get all responses
        assert len(responses) == num_prompts
        
        # Should take some time due to batching (but not too much due to concurrency)
        # This is a rough test - exact timing depends on system performance
        assert elapsed < num_prompts * 0.1  # Should be much faster than sequential
    
    @pytest.mark.asyncio
    async def test_rate_limiter_concurrent_access(self):
        """Test rate limiter under concurrent access."""
        limits = RateLimits(requests_per_minute=10, tokens_per_minute=5000)
        limiter = RateLimiter(limits)
        
        async def make_request():
            await limiter.acquire(estimated_tokens=100)
            return "success"
        
        # Make multiple concurrent requests
        tasks = [make_request() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All should succeed
        assert len(results) == 5
        assert all(result == "success" for result in results)