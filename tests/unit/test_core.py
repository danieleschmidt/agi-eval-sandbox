"""
Core functionality tests for AGI Evaluation Sandbox.
"""

import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from agi_eval_sandbox.core import EvalSuite, Model
from agi_eval_sandbox.core.benchmarks import TruthfulQABenchmark, Question, QuestionType, Score
from agi_eval_sandbox.core.models import LocalProvider, ModelConfig


class TestModels:
    """Test model functionality."""
    
    def test_local_provider_creation(self):
        """Test local provider creation."""
        config = ModelConfig(name="test-model", provider="local")
        provider = LocalProvider(config)
        assert provider.config.name == "test-model"
        assert provider.config.provider == "local"
    
    @pytest.mark.asyncio
    async def test_local_provider_generation(self):
        """Test local provider response generation."""
        config = ModelConfig(name="test-model", provider="local")
        provider = LocalProvider(config)
        
        response = await provider.generate("Test prompt")
        assert "Mock response for: Test prompt" in response
        
        responses = await provider.batch_generate(["Prompt 1", "Prompt 2"])
        assert len(responses) == 2
        assert all("Mock response" in r for r in responses)
    
    def test_model_creation(self):
        """Test Model class creation."""
        model = Model(provider="local", name="test-model")
        assert model.name == "test-model"
        assert model.provider_name == "local"
        
        limits = model.get_limits()
        assert limits.requests_per_minute == 1000  # Local provider limits


class TestEvalSuite:
    """Test evaluation suite functionality."""
    
    def test_eval_suite_creation(self):
        """Test EvalSuite creation and benchmark registration."""
        suite = EvalSuite()
        benchmarks = suite.list_benchmarks()
        
        # Should have default benchmarks
        expected_benchmarks = ["truthfulqa", "mmlu", "humaneval"]
        for benchmark in expected_benchmarks:
            assert benchmark in benchmarks
    
    def test_benchmark_retrieval(self):
        """Test benchmark retrieval."""
        suite = EvalSuite()
        
        benchmark = suite.get_benchmark("truthfulqa")
        assert benchmark is not None
        assert benchmark.name == "truthfulqa"
        
        # Non-existent benchmark
        benchmark = suite.get_benchmark("nonexistent")
        assert benchmark is None
    
    @pytest.mark.asyncio
    async def test_simple_evaluation(self):
        """Test simple evaluation with local model."""
        suite = EvalSuite()
        model = Model(provider="local", name="test-model")
        
        # Run evaluation with limited questions
        results = await suite.evaluate(
            model=model,
            benchmarks=["truthfulqa"],
            num_questions=2,
            save_results=False
        )
        
        assert results is not None
        summary = results.summary()
        
        assert summary["total_benchmarks"] == 1
        assert summary["total_questions"] == 2
        assert "truthfulqa" in summary["benchmark_scores"]
        
        # Check benchmark result details
        truthfulqa_result = results.get_benchmark_result("truthfulqa")
        assert truthfulqa_result is not None
        assert truthfulqa_result.model_name == "test-model"
        assert truthfulqa_result.total_questions == 2


class TestResults:
    """Test results functionality."""
    
    def test_results_summary(self):
        """Test results summary generation."""
        from agi_eval_sandbox.core.results import Results, BenchmarkResult, EvaluationResult
        
        results = Results()
        
        # Create mock evaluation results
        eval_results = [
            EvaluationResult(
                question_id="q1",
                question_prompt="Test question 1",
                model_response="Test response 1",
                score=Score(value=1.0, passed=True),
                benchmark_name="test_benchmark"
            ),
            EvaluationResult(
                question_id="q2", 
                question_prompt="Test question 2",
                model_response="Test response 2",
                score=Score(value=0.5, passed=False),
                benchmark_name="test_benchmark"
            )
        ]
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test_benchmark",
            model_name="test_model",
            model_provider="local",
            results=eval_results
        )
        
        results.add_benchmark_result(benchmark_result)
        
        summary = results.summary()
        assert summary["total_benchmarks"] == 1
        assert summary["total_questions"] == 2
        assert summary["overall_score"] == 0.75  # (1.0 + 0.5) / 2
        assert summary["overall_pass_rate"] == 50.0  # 1 passed out of 2
    
    def test_results_dataframe_conversion(self):
        """Test results to DataFrame conversion."""
        from agi_eval_sandbox.core.results import Results, BenchmarkResult, EvaluationResult
        
        results = Results()
        
        eval_result = EvaluationResult(
            question_id="q1",
            question_prompt="Test question",
            model_response="Test response",
            score=Score(value=0.8, passed=True),
            benchmark_name="test_benchmark"
        )
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test_benchmark",
            model_name="test_model", 
            model_provider="local",
            results=[eval_result]
        )
        
        results.add_benchmark_result(benchmark_result)
        
        df = results.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["benchmark"] == "test_benchmark"
        assert df.iloc[0]["model_name"] == "test_model"
        assert df.iloc[0]["score"] == 0.8
        assert df.iloc[0]["passed"] is True


@pytest.mark.asyncio 
async def test_integration_evaluation_flow():
    """Integration test for complete evaluation flow."""
    suite = EvalSuite()
    model = Model(provider="local", name="integration-test-model")
    
    # Run evaluation
    results = await suite.evaluate(
        model=model,
        benchmarks=["truthfulqa"],
        num_questions=1,
        save_results=False
    )
    
    # Verify results structure
    assert results is not None
    summary = results.summary()
    
    assert summary["total_benchmarks"] == 1 
    assert summary["total_questions"] == 1
    assert 0.0 <= summary["overall_score"] <= 1.0
    assert 0.0 <= summary["overall_pass_rate"] <= 100.0
    
    # Test export functionality
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as temp_dir:
        json_path = os.path.join(temp_dir, "results.json")
        results.export("json", json_path)
        assert os.path.exists(json_path)
        
        csv_path = os.path.join(temp_dir, "results.csv")
        results.export("csv", csv_path)
        assert os.path.exists(csv_path)