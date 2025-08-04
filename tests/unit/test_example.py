"""
Core functionality tests for AGI Evaluation Sandbox.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from agi_eval_sandbox.core import EvalSuite, Model
from agi_eval_sandbox.core.benchmarks import TruthfulQABenchmark, Question, QuestionType, Score
from agi_eval_sandbox.core.models import LocalProvider, ModelConfig


class TestBenchmarks:
    """Test benchmark functionality."""
    
    def test_truthfulqa_benchmark_creation(self):
        """Test TruthfulQA benchmark creation."""
        benchmark = TruthfulQABenchmark()
        assert benchmark.name == "truthfulqa"
        assert benchmark.version == "1.0"
        
        questions = benchmark.get_questions()
        assert len(questions) == 3
        assert all(isinstance(q, Question) for q in questions)
    
    def test_truthfulqa_evaluation(self):
        """Test TruthfulQA evaluation logic."""
        benchmark = TruthfulQABenchmark()
        questions = benchmark.get_questions()
        
        # Test first question (watermelon seeds)
        q1 = questions[0]
        
        # Correct answer
        correct_response = "Nothing happens if you eat watermelon seeds. They are safe."
        score = benchmark.evaluate_response(q1, correct_response)
        assert score.passed is True
        assert score.value == 1.0
        
        # Incorrect answer
        wrong_response = "You will die if you eat watermelon seeds."
        score = benchmark.evaluate_response(q1, wrong_response)
        assert score.passed is False
        assert score.value == 0.0


class TestEvaluationEngine:
    """Unit tests for the evaluation engine."""
    
    def test_create_evaluation(self, sample_evaluation_config):
        """Test creating a new evaluation."""
        # This would test your evaluation creation logic
        assert sample_evaluation_config["model"]["provider"] == "openai"
        assert "mmlu" in sample_evaluation_config["benchmarks"]
    
    def test_evaluation_validation(self):
        """Test evaluation configuration validation."""
        # Test invalid configuration
        invalid_config = {
            "model": {},  # Missing required fields
            "benchmarks": []  # Empty benchmarks
        }
        
        # This would use your actual validation logic
        # with pytest.raises(ValidationError):
        #     validate_evaluation_config(invalid_config)
        
        # For now, just test the structure
        assert "model" in invalid_config
        assert "benchmarks" in invalid_config
    
    @pytest.mark.asyncio
    async def test_async_evaluation_execution(self, mock_async_model_provider):
        """Test async evaluation execution."""
        # Test async evaluation logic
        result = await mock_async_model_provider.generate("Test prompt")
        assert "Generated response for: Test prompt" in result
    
    def test_batch_evaluation(self, mock_model_provider):
        """Test batch evaluation processing."""
        prompts = ["Prompt 1", "Prompt 2", "Prompt 3"]
        results = mock_model_provider.batch_generate(prompts)
        
        assert len(results) == len(prompts)
        assert all(isinstance(result, str) for result in results)


class TestBenchmarkProcessing:
    """Unit tests for benchmark processing."""
    
    def test_question_loading(self, sample_benchmark_questions):
        """Test loading benchmark questions."""
        questions = sample_benchmark_questions
        
        assert len(questions) == 2
        assert questions[0]["benchmark"] == "mmlu"
        assert questions[1]["benchmark"] == "truthfulqa"
    
    def test_answer_evaluation(self):
        """Test answer evaluation logic."""
        # Test exact match
        correct_answer = "299,792,458 m/s"
        model_response = "299,792,458 m/s"
        
        # This would use your actual evaluation logic
        # is_correct = evaluate_answer(correct_answer, model_response)
        is_correct = correct_answer == model_response
        
        assert is_correct is True
    
    def test_scoring_calculation(self, sample_evaluation_results):
        """Test scoring calculation."""
        results = sample_evaluation_results
        
        assert results["metrics"]["accuracy"] == 0.85
        assert results["metrics"]["total_questions"] == 100
        assert results["metrics"]["correct_answers"] == 85


class TestModelProviders:
    """Unit tests for model provider integrations."""
    
    def test_openai_provider_initialization(self, mock_env_vars):
        """Test OpenAI provider initialization."""
        # This would test your actual provider initialization
        api_key = mock_env_vars["OPENAI_API_KEY"]
        assert api_key == "test-openai-key"
    
    def test_anthropic_provider_initialization(self, mock_env_vars):
        """Test Anthropic provider initialization."""
        api_key = mock_env_vars["ANTHROPIC_API_KEY"]
        assert api_key == "test-anthropic-key"
    
    def test_provider_rate_limiting(self, mock_model_provider):
        """Test provider rate limiting."""
        limits = mock_model_provider.get_limits()
        
        assert "requests_per_minute" in limits
        assert "tokens_per_minute" in limits
        assert limits["requests_per_minute"] > 0


class TestDataProcessing:
    """Unit tests for data processing utilities."""
    
    def test_result_serialization(self, sample_evaluation_results):
        """Test result serialization to JSON."""
        import json
        
        # Test that results can be serialized
        serialized = json.dumps(sample_evaluation_results)
        deserialized = json.loads(serialized)
        
        assert deserialized["evaluation_id"] == sample_evaluation_results["evaluation_id"]
        assert deserialized["metrics"]["accuracy"] == 0.85
    
    def test_data_validation(self):
        """Test data validation utilities."""
        from tests.conftest import assert_valid_uuid, assert_valid_timestamp
        
        # Test UUID validation
        valid_uuid = "550e8400-e29b-41d4-a716-446655440000"
        invalid_uuid = "not-a-uuid"
        
        assert assert_valid_uuid(valid_uuid) is True
        assert assert_valid_uuid(invalid_uuid) is False
        
        # Test timestamp validation
        valid_timestamp = "2024-01-15T10:00:00Z"
        invalid_timestamp = "not-a-timestamp"
        
        assert assert_valid_timestamp(valid_timestamp) is True
        assert assert_valid_timestamp(invalid_timestamp) is False


class TestConfigurationManagement:
    """Unit tests for configuration management."""
    
    def test_environment_loading(self, mock_env_vars):
        """Test environment variable loading."""
        assert mock_env_vars["ENVIRONMENT"] == "test"
        assert mock_env_vars["DEBUG"] == "true"
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test required configuration fields
        required_fields = [
            "DATABASE_URL",
            "REDIS_URL",
            "SECRET_KEY"
        ]
        
        # This would use your actual config validation
        for field in required_fields:
            assert field is not None  # Placeholder assertion


class TestUtilityFunctions:
    """Unit tests for utility functions."""
    
    def test_string_processing(self):
        """Test string processing utilities."""
        # Example utility function tests
        test_string = "  Hello, World!  "
        
        # Test trimming
        trimmed = test_string.strip()
        assert trimmed == "Hello, World!"
        
        # Test case conversion
        lower_case = test_string.lower().strip()
        assert lower_case == "hello, world!"
    
    def test_data_transformation(self):
        """Test data transformation utilities."""
        # Example data transformation
        input_data = {"key1": "value1", "key2": 42}
        
        # Test key transformation
        transformed_keys = {k.upper(): v for k, v in input_data.items()}
        expected = {"KEY1": "value1", "KEY2": 42}
        
        assert transformed_keys == expected
    
    def test_error_handling(self):
        """Test error handling utilities."""
        # Test error handling
        with pytest.raises(ValueError):
            raise ValueError("Test error")
        
        # Test exception catching
        try:
            raise RuntimeError("Test runtime error")
        except RuntimeError as e:
            assert str(e) == "Test runtime error"


# Parameterized tests
@pytest.mark.parametrize("benchmark_name,expected_type", [
    ("mmlu", "multiple_choice"),
    ("truthfulqa", "open_ended"),
    ("humaneval", "code_generation"),
])
def test_benchmark_types(benchmark_name, expected_type):
    """Test different benchmark types."""
    # This would use your actual benchmark type detection
    benchmark_types = {
        "mmlu": "multiple_choice",
        "truthfulqa": "open_ended",
        "humaneval": "code_generation"
    }
    
    assert benchmark_types.get(benchmark_name) == expected_type


@pytest.mark.parametrize("model_name,provider", [
    ("gpt-4", "openai"),
    ("claude-3", "anthropic"),
    ("gemini-pro", "google"),
])
def test_model_providers(model_name, provider):
    """Test model provider detection."""
    model_providers = {
        "gpt-4": "openai",
        "claude-3": "anthropic",
        "gemini-pro": "google"
    }
    
    assert model_providers.get(model_name) == provider


# Performance tests
@pytest.mark.slow
def test_large_batch_processing():
    """Test processing large batches of data."""
    import time
    
    start_time = time.time()
    
    # Simulate processing large batch
    large_batch = list(range(10000))
    processed = [x * 2 for x in large_batch]
    
    end_time = time.time()
    processing_time = end_time - start_time
    
    assert len(processed) == 10000
    assert processing_time < 1.0  # Should complete in under 1 second


# Skip tests conditionally
@pytest.mark.skip_ci
def test_local_only_functionality():
    """Test functionality that only works locally."""
    # This test would be skipped in CI environments
    pass


@pytest.mark.requires_gpu
def test_gpu_functionality():
    """Test GPU-dependent functionality."""
    # This test would be skipped if no GPU is available
    pass


@pytest.mark.requires_model_api
def test_model_api_integration():
    """Test integration with model APIs."""
    # This test would be skipped if API keys are not available
    pass