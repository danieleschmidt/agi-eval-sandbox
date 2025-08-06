"""
Comprehensive unit tests for results management functionality.
"""

import pytest
import json
import tempfile
import os
import statistics
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

from agi_eval_sandbox.core.results import (
    EvaluationResult, BenchmarkResult, Results
)
from agi_eval_sandbox.core.benchmarks import Score, Question, QuestionType


@pytest.mark.unit
class TestEvaluationResult:
    """Test EvaluationResult dataclass functionality."""
    
    def test_evaluation_result_creation(self):
        """Test basic EvaluationResult creation."""
        score = Score(value=0.85, passed=True, explanation="Good answer")
        
        result = EvaluationResult(
            question_id="test-q1",
            question_prompt="What is 2+2?",
            model_response="4",
            score=score,
            benchmark_name="math_test",
            category="arithmetic",
            metadata={"confidence": 0.9}
        )
        
        assert result.question_id == "test-q1"
        assert result.question_prompt == "What is 2+2?"
        assert result.model_response == "4"
        assert result.score == score
        assert result.benchmark_name == "math_test"
        assert result.category == "arithmetic"
        assert result.metadata["confidence"] == 0.9
        assert isinstance(result.timestamp, datetime)
    
    def test_evaluation_result_defaults(self):
        """Test EvaluationResult with default values."""
        score = Score(value=0.5, passed=False)
        
        result = EvaluationResult(
            question_id="minimal",
            question_prompt="Minimal question",
            model_response="Minimal response",
            score=score,
            benchmark_name="minimal_test"
        )
        
        assert result.category is None
        assert isinstance(result.timestamp, datetime)
        assert isinstance(result.metadata, dict)
        assert len(result.metadata) == 0
    
    def test_evaluation_result_timestamp_auto(self):
        """Test that timestamp is automatically set."""
        score = Score(value=1.0, passed=True)
        
        before = datetime.now()
        result = EvaluationResult(
            question_id="time-test",
            question_prompt="Time test",
            model_response="Response",
            score=score,
            benchmark_name="time_test"
        )
        after = datetime.now()
        
        assert before <= result.timestamp <= after


@pytest.mark.unit
class TestBenchmarkResult:
    """Test BenchmarkResult functionality."""
    
    def create_sample_evaluation_results(self, count=5):
        """Helper to create sample evaluation results."""
        results = []
        for i in range(count):
            score_value = 0.2 * i  # 0.0, 0.2, 0.4, 0.6, 0.8
            score = Score(
                value=score_value,
                passed=score_value >= 0.5,
                explanation=f"Result {i}"
            )
            
            result = EvaluationResult(
                question_id=f"q{i}",
                question_prompt=f"Question {i}",
                model_response=f"Response {i}",
                score=score,
                benchmark_name="test_benchmark",
                category="math" if i % 2 == 0 else "science"
            )
            results.append(result)
        
        return results
    
    def test_benchmark_result_creation(self):
        """Test BenchmarkResult creation."""
        eval_results = self.create_sample_evaluation_results(3)
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test_benchmark",
            model_name="test_model",
            model_provider="test_provider",
            results=eval_results,
            config={"temperature": 0.7}
        )
        
        assert benchmark_result.benchmark_name == "test_benchmark"
        assert benchmark_result.model_name == "test_model"
        assert benchmark_result.model_provider == "test_provider"
        assert len(benchmark_result.results) == 3
        assert benchmark_result.config["temperature"] == 0.7
        assert isinstance(benchmark_result.timestamp, datetime)
    
    def test_benchmark_result_total_questions(self):
        """Test total_questions property."""
        eval_results = self.create_sample_evaluation_results(7)
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test",
            model_name="test",
            model_provider="test",
            results=eval_results
        )
        
        assert benchmark_result.total_questions == 7
    
    def test_benchmark_result_empty_results(self):
        """Test BenchmarkResult with empty results."""
        benchmark_result = BenchmarkResult(
            benchmark_name="empty_test",
            model_name="test_model",
            model_provider="test_provider",
            results=[]
        )
        
        assert benchmark_result.total_questions == 0
        assert benchmark_result.average_score == 0.0
        assert benchmark_result.pass_rate == 0.0
        assert len(benchmark_result.category_scores) == 0
    
    def test_benchmark_result_average_score(self):
        """Test average_score property."""
        eval_results = self.create_sample_evaluation_results(5)
        # Scores are 0.0, 0.2, 0.4, 0.6, 0.8
        # Average should be 0.4
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test",
            model_name="test",
            model_provider="test",
            results=eval_results
        )
        
        expected_average = statistics.mean([0.0, 0.2, 0.4, 0.6, 0.8])
        assert abs(benchmark_result.average_score - expected_average) < 0.001
    
    def test_benchmark_result_pass_rate(self):
        """Test pass_rate property."""
        eval_results = self.create_sample_evaluation_results(5)
        # Passed scores: 0.6 (>=0.5) and 0.8 (>=0.5) = 2 out of 5 = 40%
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test",
            model_name="test", 
            model_provider="test",
            results=eval_results
        )
        
        assert benchmark_result.pass_rate == 40.0
    
    def test_benchmark_result_category_scores(self):
        """Test category_scores property."""
        eval_results = self.create_sample_evaluation_results(5)
        # Even indices (0,2,4) -> math: scores 0.0, 0.4, 0.8 -> avg 0.4
        # Odd indices (1,3) -> science: scores 0.2, 0.6 -> avg 0.4
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test",
            model_name="test",
            model_provider="test", 
            results=eval_results
        )
        
        category_scores = benchmark_result.category_scores
        
        assert "math" in category_scores
        assert "science" in category_scores
        
        # Math: (0.0 + 0.4 + 0.8) / 3 = 0.4
        assert abs(category_scores["math"] - 0.4) < 0.001
        
        # Science: (0.2 + 0.6) / 2 = 0.4  
        assert abs(category_scores["science"] - 0.4) < 0.001
    
    def test_benchmark_result_category_scores_uncategorized(self):
        """Test category_scores with uncategorized questions."""
        # Create results without categories
        eval_results = []
        for i in range(3):
            score = Score(value=0.5, passed=True)
            result = EvaluationResult(
                question_id=f"uncategorized-{i}",
                question_prompt=f"Question {i}",
                model_response=f"Response {i}",
                score=score,
                benchmark_name="test",
                category=None  # No category
            )
            eval_results.append(result)
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test",
            model_name="test",
            model_provider="test",
            results=eval_results
        )
        
        category_scores = benchmark_result.category_scores
        assert "uncategorized" in category_scores
        assert category_scores["uncategorized"] == 0.5
    
    def test_benchmark_result_get_failed_questions(self):
        """Test get_failed_questions method."""
        eval_results = self.create_sample_evaluation_results(5)
        # Failed scores: 0.0, 0.2, 0.4 (all < 0.5)
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test",
            model_name="test",
            model_provider="test",
            results=eval_results
        )
        
        failed_questions = benchmark_result.get_failed_questions()
        
        assert len(failed_questions) == 3
        assert all(not result.score.passed for result in failed_questions)
        assert [result.question_id for result in failed_questions] == ["q0", "q1", "q2"]
    
    def test_benchmark_result_get_results_by_category(self):
        """Test get_results_by_category method."""
        eval_results = self.create_sample_evaluation_results(5)
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test",
            model_name="test",
            model_provider="test",
            results=eval_results
        )
        
        math_results = benchmark_result.get_results_by_category("math")
        science_results = benchmark_result.get_results_by_category("science")
        
        assert len(math_results) == 3  # q0, q2, q4
        assert len(science_results) == 2  # q1, q3
        
        assert all(result.category == "math" for result in math_results)
        assert all(result.category == "science" for result in science_results)
    
    def test_benchmark_result_get_results_by_nonexistent_category(self):
        """Test get_results_by_category with non-existent category."""
        eval_results = self.create_sample_evaluation_results(3)
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test",
            model_name="test",
            model_provider="test",
            results=eval_results
        )
        
        nonexistent_results = benchmark_result.get_results_by_category("nonexistent")
        assert len(nonexistent_results) == 0


@pytest.mark.unit
class TestResults:
    """Test Results container functionality."""
    
    def create_sample_benchmark_result(self, benchmark_name="test", model_name="test_model"):
        """Helper to create sample benchmark result."""
        eval_results = []
        for i in range(3):
            score = Score(value=0.7, passed=True)
            result = EvaluationResult(
                question_id=f"{benchmark_name}-q{i}",
                question_prompt=f"Question {i}",
                model_response=f"Response {i}",
                score=score,
                benchmark_name=benchmark_name
            )
            eval_results.append(result)
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            model_name=model_name,
            model_provider="test_provider",
            results=eval_results
        )
    
    def test_results_creation(self):
        """Test Results creation."""
        results = Results()
        
        assert len(results.benchmark_results) == 0
        assert isinstance(results.run_id, str)
        assert len(results.run_id) > 0
        assert isinstance(results.timestamp, datetime)
        assert isinstance(results.metadata, dict)
    
    def test_results_add_benchmark_result(self):
        """Test adding benchmark results."""
        results = Results()
        
        benchmark_result1 = self.create_sample_benchmark_result("benchmark1")
        benchmark_result2 = self.create_sample_benchmark_result("benchmark2")
        
        results.add_benchmark_result(benchmark_result1)
        results.add_benchmark_result(benchmark_result2)
        
        assert len(results.benchmark_results) == 2
        assert results.benchmark_results[0] == benchmark_result1
        assert results.benchmark_results[1] == benchmark_result2
    
    def test_results_get_benchmark_result(self):
        """Test getting specific benchmark result."""
        results = Results()
        
        benchmark_result = self.create_sample_benchmark_result("target_benchmark")
        results.add_benchmark_result(benchmark_result)
        
        retrieved = results.get_benchmark_result("target_benchmark")
        assert retrieved == benchmark_result
        
        # Non-existent benchmark
        non_existent = results.get_benchmark_result("non_existent")
        assert non_existent is None
    
    def test_results_summary_empty(self):
        """Test summary with no results."""
        results = Results()
        
        summary = results.summary()
        
        assert summary["run_id"] == results.run_id
        assert summary["total_benchmarks"] == 0
        assert summary["total_questions"] == 0
        assert summary["overall_score"] == 0.0
        assert summary["overall_pass_rate"] == 0.0
        assert len(summary["benchmark_scores"]) == 0
    
    def test_results_summary_with_data(self):
        """Test summary with benchmark results."""
        results = Results()
        
        # Add benchmark with known scores
        benchmark_result = self.create_sample_benchmark_result("test_benchmark")
        results.add_benchmark_result(benchmark_result)
        
        summary = results.summary()
        
        assert summary["total_benchmarks"] == 1
        assert summary["total_questions"] == 3
        assert summary["overall_score"] == 0.7  # All scores are 0.7
        assert summary["overall_pass_rate"] == 100.0  # All passed
        assert "test_benchmark" in summary["benchmark_scores"]
        assert summary["benchmark_scores"]["test_benchmark"] == 0.7
    
    def test_results_summary_multiple_benchmarks(self):
        """Test summary with multiple benchmarks."""
        results = Results()
        
        # Create benchmarks with different scores
        eval_results1 = [
            EvaluationResult(
                question_id="q1",
                question_prompt="Q1",
                model_response="R1",
                score=Score(value=1.0, passed=True),
                benchmark_name="high_score"
            )
        ]
        
        eval_results2 = [
            EvaluationResult(
                question_id="q2",
                question_prompt="Q2", 
                model_response="R2",
                score=Score(value=0.0, passed=False),
                benchmark_name="low_score"
            )
        ]
        
        benchmark1 = BenchmarkResult("high_score", "model", "provider", eval_results1)
        benchmark2 = BenchmarkResult("low_score", "model", "provider", eval_results2)
        
        results.add_benchmark_result(benchmark1)
        results.add_benchmark_result(benchmark2)
        
        summary = results.summary()
        
        assert summary["total_benchmarks"] == 2
        assert summary["total_questions"] == 2
        assert summary["overall_score"] == 0.5  # (1.0 + 0.0) / 2
        assert summary["overall_pass_rate"] == 50.0  # 1 passed out of 2
        assert summary["benchmark_scores"]["high_score"] == 1.0
        assert summary["benchmark_scores"]["low_score"] == 0.0
    
    def test_results_to_dataframe(self):
        """Test conversion to DataFrame."""
        results = Results()
        
        # Add some test data
        benchmark_result = self.create_sample_benchmark_result("df_test")
        results.add_benchmark_result(benchmark_result)
        
        df = results.to_dataframe()
        
        # Check DataFrame structure
        assert len(df) == 3  # 3 evaluation results
        
        expected_columns = [
            "run_id", "benchmark", "model_name", "model_provider",
            "question_id", "question_prompt", "model_response",
            "score", "passed", "category", "timestamp"
        ]
        
        for col in expected_columns:
            assert col in df.columns
        
        # Check data values
        assert all(df["benchmark"] == "df_test")
        assert all(df["model_name"] == "test_model")
        assert all(df["score"] == 0.7)
        assert all(df["passed"] == True)
    
    def test_results_to_dataframe_empty(self):
        """Test DataFrame conversion with no results."""
        results = Results()
        
        df = results.to_dataframe()
        
        assert len(df) == 0
        assert len(df.columns) > 0  # Should still have column headers
    
    def test_results_export_json(self):
        """Test JSON export functionality."""
        results = Results()
        results.metadata["test_key"] = "test_value"
        
        benchmark_result = self.create_sample_benchmark_result("export_test")
        results.add_benchmark_result(benchmark_result)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path = os.path.join(temp_dir, "test_results.json")
            
            results.export("json", json_path)
            
            assert os.path.exists(json_path)
            
            # Verify JSON content
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            assert data["run_id"] == results.run_id
            assert data["metadata"]["test_key"] == "test_value"
            assert len(data["benchmark_results"]) == 1
            assert data["benchmark_results"][0]["benchmark_name"] == "export_test"
    
    def test_results_export_csv(self):
        """Test CSV export functionality."""
        results = Results()
        
        benchmark_result = self.create_sample_benchmark_result("csv_test")
        results.add_benchmark_result(benchmark_result)
        
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = os.path.join(temp_dir, "test_results.csv")
            
            results.export("csv", csv_path)
            
            assert os.path.exists(csv_path)
            
            # Verify CSV has content
            with open(csv_path, 'r') as f:
                content = f.read()
            
            assert "benchmark" in content  # Header
            assert "csv_test" in content  # Data
            assert "test_model" in content
    
    def test_results_export_unsupported_format(self):
        """Test export with unsupported format."""
        results = Results()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_path = os.path.join(temp_dir, "test.xml")
            
            with pytest.raises(ValueError, match="Unsupported export format"):
                results.export("xml", invalid_path)
    
    def test_results_compare_with(self):
        """Test comparison functionality between result sets."""
        results1 = Results()
        results2 = Results()
        
        # Create results for same benchmark but different models
        eval_result1 = EvaluationResult(
            question_id="q1",
            question_prompt="Question 1",
            model_response="Response 1",
            score=Score(value=0.8, passed=True),
            benchmark_name="compare_test"
        )
        
        eval_result2 = EvaluationResult(
            question_id="q1", 
            question_prompt="Question 1",
            model_response="Response 1",
            score=Score(value=0.6, passed=True),
            benchmark_name="compare_test"
        )
        
        benchmark1 = BenchmarkResult("compare_test", "model1", "provider", [eval_result1])
        benchmark2 = BenchmarkResult("compare_test", "model2", "provider", [eval_result2])
        
        results1.add_benchmark_result(benchmark1)
        results2.add_benchmark_result(benchmark2)
        
        comparison = results1.compare_with(results2)
        
        assert "model1_vs_model2" in comparison
        assert "compare_test" in comparison["model1_vs_model2"]
        assert comparison["model1_vs_model2"]["compare_test"]["score_difference"] == 0.2  # 0.8 - 0.6
        assert comparison["model1_vs_model2"]["compare_test"]["model1_better"] is True
    
    def test_results_get_best_performing_model(self):
        """Test getting best performing model."""
        results = Results()
        
        # Create multiple benchmark results with different scores
        models_scores = [("model_a", 0.9), ("model_b", 0.7), ("model_c", 0.95)]
        
        for model_name, score in models_scores:
            eval_result = EvaluationResult(
                question_id="q1",
                question_prompt="Test",
                model_response="Test",
                score=Score(value=score, passed=True),
                benchmark_name="performance_test"
            )
            
            benchmark_result = BenchmarkResult(
                "performance_test", model_name, "provider", [eval_result]
            )
            results.add_benchmark_result(benchmark_result)
        
        best_model = results.get_best_performing_model()
        
        assert best_model["model_name"] == "model_c"
        assert best_model["average_score"] == 0.95
    
    def test_results_get_worst_performing_model(self):
        """Test getting worst performing model."""
        results = Results()
        
        # Create multiple benchmark results with different scores
        models_scores = [("model_a", 0.9), ("model_b", 0.3), ("model_c", 0.7)]
        
        for model_name, score in models_scores:
            eval_result = EvaluationResult(
                question_id="q1",
                question_prompt="Test",
                model_response="Test", 
                score=Score(value=score, passed=score >= 0.5),
                benchmark_name="performance_test"
            )
            
            benchmark_result = BenchmarkResult(
                "performance_test", model_name, "provider", [eval_result]
            )
            results.add_benchmark_result(benchmark_result)
        
        worst_model = results.get_worst_performing_model()
        
        assert worst_model["model_name"] == "model_b"
        assert worst_model["average_score"] == 0.3
    
    def test_results_filter_by_benchmark(self):
        """Test filtering results by benchmark."""
        results = Results()
        
        # Add results for different benchmarks
        benchmark1 = self.create_sample_benchmark_result("benchmark_a")
        benchmark2 = self.create_sample_benchmark_result("benchmark_b")
        
        results.add_benchmark_result(benchmark1)
        results.add_benchmark_result(benchmark2)
        
        filtered_results = results.filter_by_benchmark("benchmark_a")
        
        assert len(filtered_results.benchmark_results) == 1
        assert filtered_results.benchmark_results[0].benchmark_name == "benchmark_a"
    
    def test_results_filter_by_model(self):
        """Test filtering results by model."""
        results = Results()
        
        # Add results for different models
        benchmark1 = self.create_sample_benchmark_result("test_benchmark", "model_x")
        benchmark2 = self.create_sample_benchmark_result("test_benchmark", "model_y")
        
        results.add_benchmark_result(benchmark1)
        results.add_benchmark_result(benchmark2)
        
        filtered_results = results.filter_by_model("model_x")
        
        assert len(filtered_results.benchmark_results) == 1
        assert filtered_results.benchmark_results[0].model_name == "model_x"


@pytest.mark.unit
class TestResultsEdgeCases:
    """Test edge cases and boundary conditions in results."""
    
    def test_results_with_extreme_scores(self):
        """Test results handling with extreme score values."""
        results = Results()
        
        # Create results with extreme scores
        extreme_results = []
        extreme_values = [0.0, 0.000001, 0.999999, 1.0, -0.5, 1.5]  # Including invalid ranges
        
        for i, value in enumerate(extreme_values):
            score = Score(value=value, passed=value >= 0.5)
            result = EvaluationResult(
                question_id=f"extreme-{i}",
                question_prompt=f"Extreme question {i}",
                model_response=f"Response {i}",
                score=score,
                benchmark_name="extreme_test"
            )
            extreme_results.append(result)
        
        benchmark_result = BenchmarkResult(
            "extreme_test", "test_model", "test_provider", extreme_results
        )
        
        results.add_benchmark_result(benchmark_result)
        
        summary = results.summary()
        
        # Should handle extreme values gracefully
        assert isinstance(summary["overall_score"], (int, float))
        assert isinstance(summary["overall_pass_rate"], (int, float))
        assert 0.0 <= summary["overall_pass_rate"] <= 100.0
    
    def test_results_with_very_long_strings(self):
        """Test results with very long string values."""
        results = Results()
        
        long_prompt = "A" * 10000
        long_response = "B" * 10000
        
        score = Score(value=0.5, passed=True)
        eval_result = EvaluationResult(
            question_id="long-string-test",
            question_prompt=long_prompt,
            model_response=long_response,
            score=score,
            benchmark_name="long_string_test"
        )
        
        benchmark_result = BenchmarkResult(
            "long_string_test", "test_model", "test_provider", [eval_result]
        )
        
        results.add_benchmark_result(benchmark_result)
        
        # Should handle long strings without issues
        summary = results.summary()
        assert summary["total_questions"] == 1
        
        df = results.to_dataframe()
        assert len(df) == 1
        assert len(df.iloc[0]["question_prompt"]) == 10000
        assert len(df.iloc[0]["model_response"]) == 10000
    
    def test_results_with_unicode_content(self):
        """Test results with Unicode characters."""
        results = Results()
        
        unicode_prompt = "What does 世界 mean? 🌍"
        unicode_response = "It means 'world' in Chinese 中文"
        
        score = Score(value=0.9, passed=True)
        eval_result = EvaluationResult(
            question_id="unicode-test",
            question_prompt=unicode_prompt,
            model_response=unicode_response,
            score=score,
            benchmark_name="unicode_test"
        )
        
        benchmark_result = BenchmarkResult(
            "unicode_test", "test_model", "test_provider", [eval_result]
        )
        
        results.add_benchmark_result(benchmark_result)
        
        # Should handle Unicode correctly
        df = results.to_dataframe()
        assert "世界" in df.iloc[0]["question_prompt"]
        assert "🌍" in df.iloc[0]["question_prompt"]
        assert "中文" in df.iloc[0]["model_response"]
    
    def test_results_concurrent_modification(self):
        """Test results behavior under concurrent-like modifications."""
        results = Results()
        
        # Simulate adding results while iterating (common concurrency issue)
        benchmark_result = Results().create_sample_benchmark_result("concurrent_test")
        results.add_benchmark_result(benchmark_result)
        
        # Get snapshot of results
        initial_count = len(results.benchmark_results)
        
        # Add more results
        for i in range(5):
            new_result = Results().create_sample_benchmark_result(f"concurrent_{i}")
            results.add_benchmark_result(new_result)
        
        final_count = len(results.benchmark_results)
        assert final_count == initial_count + 5
    
    def test_results_memory_efficiency(self):
        """Test results memory handling with large datasets."""
        results = Results()
        
        # Create a large number of evaluation results
        large_eval_results = []
        for i in range(1000):
            score = Score(value=0.5 + (i % 2) * 0.3, passed=True)  # Alternate between 0.5 and 0.8
            result = EvaluationResult(
                question_id=f"memory-test-{i}",
                question_prompt=f"Memory test question {i}",
                model_response=f"Memory test response {i}",
                score=score,
                benchmark_name="memory_test"
            )
            large_eval_results.append(result)
        
        benchmark_result = BenchmarkResult(
            "memory_test", "memory_model", "memory_provider", large_eval_results
        )
        
        results.add_benchmark_result(benchmark_result)
        
        # Should handle large datasets efficiently
        summary = results.summary()
        assert summary["total_questions"] == 1000
        
        # DataFrame conversion should work
        df = results.to_dataframe()
        assert len(df) == 1000
    
    def test_results_with_missing_data(self):
        """Test results handling with missing or None data."""
        results = Results()
        
        # Create evaluation result with some None values
        score = Score(value=0.7, passed=True)
        eval_result = EvaluationResult(
            question_id="missing-data-test",
            question_prompt="Test question",
            model_response="",  # Empty response
            score=score,
            benchmark_name="missing_data_test",
            category=None  # No category
        )
        
        benchmark_result = BenchmarkResult(
            "missing_data_test", "test_model", "test_provider", [eval_result],
            config={}  # Empty config
        )
        
        results.add_benchmark_result(benchmark_result)
        
        # Should handle missing data gracefully
        summary = results.summary()
        assert summary["total_questions"] == 1
        
        df = results.to_dataframe()
        assert len(df) == 1
        assert df.iloc[0]["model_response"] == ""
        assert df.iloc[0]["category"] is None or pd.isna(df.iloc[0]["category"])
    
    def test_results_timestamp_ordering(self):
        """Test that results maintain proper timestamp ordering."""
        results = Results()
        
        # Create results with specific timestamps
        timestamps = [
            datetime.now() - timedelta(hours=2),
            datetime.now() - timedelta(hours=1), 
            datetime.now()
        ]
        
        for i, ts in enumerate(timestamps):
            score = Score(value=0.6, passed=True)
            eval_result = EvaluationResult(
                question_id=f"timestamp-{i}",
                question_prompt=f"Timestamp test {i}",
                model_response=f"Response {i}",
                score=score,
                benchmark_name="timestamp_test"
            )
            eval_result.timestamp = ts  # Override auto-generated timestamp
            
            benchmark_result = BenchmarkResult(
                "timestamp_test", f"model_{i}", "provider", [eval_result]
            )
            benchmark_result.timestamp = ts
            
            results.add_benchmark_result(benchmark_result)
        
        # Verify timestamps are preserved
        for i, benchmark_result in enumerate(results.benchmark_results):
            assert benchmark_result.timestamp == timestamps[i]
            assert benchmark_result.results[0].timestamp == timestamps[i]
    
    def create_sample_benchmark_result(self, benchmark_name="test", model_name="test_model"):
        """Helper method to create sample benchmark result."""
        eval_results = []
        for i in range(3):
            score = Score(value=0.7, passed=True)
            result = EvaluationResult(
                question_id=f"{benchmark_name}-q{i}",
                question_prompt=f"Question {i}",
                model_response=f"Response {i}",
                score=score,
                benchmark_name=benchmark_name
            )
            eval_results.append(result)
        
        return BenchmarkResult(
            benchmark_name=benchmark_name,
            model_name=model_name,
            model_provider="test_provider",
            results=eval_results
        )