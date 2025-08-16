"""
Comprehensive tests for research framework validation.

Tests statistical rigor, reproducibility, and algorithmic correctness.
"""

import pytest
import numpy as np
import asyncio
from unittest.mock import Mock, patch
from typing import List, Dict, Any

from src.agi_eval_sandbox.research.research_framework import (
    ResearchFramework, ExperimentConfig, StatisticalAnalyzer,
    StatisticalResult, ExperimentResult
)
from src.agi_eval_sandbox.research.quantum_evaluator import QuantumInspiredEvaluator
from src.agi_eval_sandbox.research.adaptive_benchmark import AdaptiveBenchmarkSelector
from src.agi_eval_sandbox.research.neural_cache import NeuralPredictiveCache
from src.agi_eval_sandbox.core.models import Model
from src.agi_eval_sandbox.core.benchmarks import Benchmark


class TestStatisticalAnalyzer:
    """Test statistical analysis methods."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = StatisticalAnalyzer(confidence_level=0.95)
        
        # Create test samples with known properties
        np.random.seed(42)
        self.sample1 = np.random.normal(0.8, 0.1, 50).tolist()  # Higher performance
        self.sample2 = np.random.normal(0.6, 0.1, 50).tolist()  # Lower performance
        self.sample_identical = [0.7] * 50  # No variance
        
    def test_welch_t_test_significant_difference(self):
        """Test Welch's t-test with significant difference."""
        result = self.analyzer.welch_t_test(self.sample1, self.sample2)
        
        assert isinstance(result, StatisticalResult)
        assert result.is_significant == True
        assert result.p_value < 0.05
        assert abs(result.effect_size) > 0.5  # Large effect size expected
        assert len(result.confidence_interval) == 2
        assert result.sample_size_recommendation > 0
        
    def test_welch_t_test_no_difference(self):
        """Test Welch's t-test with no significant difference."""
        sample_similar = np.random.normal(0.8, 0.1, 50).tolist()
        result = self.analyzer.welch_t_test(self.sample1, sample_similar)
        
        assert result.p_value > 0.05  # Should not be significant
        assert abs(result.effect_size) < 0.5  # Small effect size
        
    def test_mann_whitney_u_test(self):
        """Test Mann-Whitney U test (non-parametric)."""
        result = self.analyzer.mann_whitney_u_test(self.sample1, self.sample2)
        
        assert isinstance(result, StatisticalResult)
        assert result.test_statistic > 0
        assert result.p_value >= 0
        assert len(result.confidence_interval) == 2
        
    def test_bootstrap_test(self):
        """Test bootstrap statistical test."""
        result = self.analyzer.bootstrap_test(
            self.sample1, self.sample2, n_bootstrap=100
        )
        
        assert isinstance(result, StatisticalResult)
        assert result.power_analysis['bootstrap_samples'] == 100
        assert len(result.confidence_interval) == 2
        
    def test_bayesian_analysis(self):
        """Test Bayesian t-test."""
        result = self.analyzer.bayesian_t_test(self.sample1, self.sample2)
        
        assert 'bayes_factor' in result
        assert 'evidence_strength' in result
        assert result['bayes_factor'] > 0
        assert result['evidence_strength'] in ['no evidence', 'weak', 'moderate', 'strong']


class TestResearchFramework:
    """Test research framework functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = ExperimentConfig(
            experiment_name="test_experiment",
            baseline_algorithms=["baseline_1"],
            novel_algorithms=["quantum_evaluator"],
            evaluation_metrics=["accuracy"],
            sample_size=20,
            confidence_level=0.95,
            random_seed=42
        )
        self.framework = ResearchFramework(self.config)
        
        # Create mock models and benchmarks
        self.models = [
            Mock(spec=Model, name="model_1"),
            Mock(spec=Model, name="model_2")
        ]
        self.benchmarks = [
            Mock(spec=Benchmark, name="benchmark_1"),
            Mock(spec=Benchmark, name="benchmark_2")
        ]
        
    @pytest.mark.asyncio
    async def test_framework_initialization(self):
        """Test framework initialization."""
        assert self.framework.config.experiment_name == "test_experiment"
        assert len(self.framework.algorithms) > 0
        assert 'quantum_evaluator' in self.framework.algorithms
        
    @pytest.mark.asyncio
    async def test_baseline_evaluation(self):
        """Test baseline algorithm evaluation."""
        results = await self.framework._evaluate_baseline_algorithms(
            self.models, self.benchmarks
        )
        
        assert isinstance(results, dict)
        assert "baseline_1" in results
        
        # Check result structure
        baseline_data = results["baseline_1"]
        assert isinstance(baseline_data, dict)
        
        # Should have results for each model-benchmark combination
        expected_keys = [f"{m.name}_{b.name}" for m in self.models for b in self.benchmarks]
        for key in expected_keys:
            assert key in baseline_data
            assert isinstance(baseline_data[key], list)
            assert len(baseline_data[key]) > 0
            
    @pytest.mark.asyncio
    async def test_novel_algorithm_evaluation(self):
        """Test novel algorithm evaluation."""
        with patch.object(
            self.framework.algorithms['quantum_evaluator'],
            'quantum_evaluate',
            return_value={
                'classical_results': {'model_1_benchmark_1': 0.8},
                'quantum_advantage': 1.2
            }
        ):
            results = await self.framework._evaluate_novel_algorithms(
                self.models, self.benchmarks
            )
            
        assert isinstance(results, dict)
        assert 'quantum_evaluator' in results
        
    @pytest.mark.asyncio
    async def test_statistical_analysis(self):
        """Test statistical analysis between algorithms."""
        # Create mock baseline and novel results
        baseline_results = {
            "baseline_1": {
                "model_1_benchmark_1": [0.6] * 10,
                "model_1_benchmark_2": [0.65] * 10
            }
        }
        
        novel_results = {
            "quantum_evaluator": {
                "model_1_benchmark_1": [0.8] * 10,
                "model_1_benchmark_2": [0.75] * 10
            }
        }
        
        comparisons = await self.framework._perform_statistical_analysis(
            baseline_results, novel_results
        )
        
        assert isinstance(comparisons, dict)
        assert "quantum_evaluator" in comparisons
        assert "baseline_1" in comparisons["quantum_evaluator"]
        
        # Check statistical results
        results = comparisons["quantum_evaluator"]["baseline_1"]
        for key, stat_result in results.items():
            assert isinstance(stat_result, StatisticalResult)
            assert hasattr(stat_result, 'p_value')
            assert hasattr(stat_result, 'effect_size')
            assert hasattr(stat_result, 'is_significant')
            
    @pytest.mark.asyncio
    async def test_reproducibility_validation(self):
        """Test reproducibility validation."""
        with patch.object(
            self.framework.algorithms['quantum_evaluator'],
            'quantum_evaluate',
            return_value={'classical_results': {'test': 0.8}}
        ):
            scores = await self.framework._validate_reproducibility(
                self.models[:1], self.benchmarks[:1]
            )
            
        assert isinstance(scores, dict)
        assert 'quantum_evaluator' in scores
        assert 0 <= scores['quantum_evaluator'] <= 1
        
    @pytest.mark.asyncio
    async def test_comprehensive_evaluation_pipeline(self):
        """Test complete evaluation pipeline."""
        with patch.object(
            self.framework.algorithms['quantum_evaluator'],
            'quantum_evaluate',
            return_value={
                'classical_results': {'model_1_benchmark_1': 0.85},
                'quantum_advantage': 1.1
            }
        ):
            report = await self.framework.run_comprehensive_evaluation(
                self.models[:1], self.benchmarks[:1]
            )
            
        # Verify report structure
        assert 'experiment_metadata' in report
        assert 'algorithm_performance' in report
        assert 'statistical_significance' in report
        assert 'reproducibility_analysis' in report
        assert 'recommendations' in report
        assert 'research_conclusions' in report
        
        # Verify metadata
        metadata = report['experiment_metadata']
        assert metadata['experiment_name'] == "test_experiment"
        assert metadata['sample_size'] == 20
        assert metadata['confidence_level'] == 0.95
        
    def test_export_research_data(self):
        """Test research data export."""
        data = self.framework.export_research_data()
        
        assert isinstance(data, dict)
        assert 'framework_name' in data
        assert 'experiment_config' in data
        assert 'statistical_methods' in data
        assert 'novel_algorithms_evaluated' in data
        assert 'research_rigor_score' in data
        
        # Verify rigor score
        assert data['research_rigor_score'] >= 0.9  # High rigor expected


class TestResearchIntegration:
    """Integration tests for research algorithms."""
    
    def setup_method(self):
        """Setup integration test fixtures."""
        self.config = ExperimentConfig(
            experiment_name="integration_test",
            baseline_algorithms=["baseline"],
            novel_algorithms=["quantum_evaluator", "adaptive_benchmark"],
            evaluation_metrics=["accuracy", "efficiency"],
            sample_size=10,  # Small for fast testing
            random_seed=42
        )
        
    @pytest.mark.asyncio
    async def test_quantum_evaluator_integration(self):
        """Test quantum evaluator integration."""
        evaluator = QuantumInspiredEvaluator(max_parallel=4, coherence_time=5.0)
        
        # Create mock models and benchmarks
        models = [Mock(spec=Model, name=f"model_{i}") for i in range(2)]
        benchmarks = [Mock(spec=Benchmark, name=f"benchmark_{i}") for i in range(2)]
        
        result = await evaluator.quantum_evaluate(
            models, benchmarks, entangle_benchmarks=True
        )
        
        assert 'classical_results' in result
        assert 'quantum_correlations' in result
        assert 'quantum_advantage' in result
        assert result['quantum_advantage'] > 0
        
    @pytest.mark.asyncio
    async def test_adaptive_benchmark_integration(self):
        """Test adaptive benchmark selector integration."""
        selector = AdaptiveBenchmarkSelector()
        
        model = Mock(spec=Model, name="test_model")
        benchmarks = [Mock(spec=Benchmark, name=f"benchmark_{i}") for i in range(5)]
        
        selected = await selector.adaptive_select(
            model, benchmarks, target_count=3
        )
        
        assert len(selected) == 3
        assert all(isinstance(b, Mock) for b in selected)
        
    @pytest.mark.asyncio
    async def test_neural_cache_integration(self):
        """Test neural cache integration."""
        cache = NeuralPredictiveCache(cache_size=100)
        
        # Test basic cache operations
        await cache.set("test_key", "test_value", {"context": "test"})
        result = await cache.get("test_key", {"context": "test"})
        
        assert result == "test_value"
        
        stats = cache.get_cache_stats()
        assert stats['cache_size'] == 1
        assert stats['max_cache_size'] == 100


class TestStatisticalRigor:
    """Test statistical rigor and correctness."""
    
    def setup_method(self):
        """Setup statistical rigor tests."""
        self.analyzer = StatisticalAnalyzer()
        
    def test_type_i_error_control(self):
        """Test Type I error rate control."""
        # Generate samples from same distribution
        np.random.seed(42)
        false_positive_count = 0
        total_tests = 100
        
        for _ in range(total_tests):
            sample1 = np.random.normal(0.5, 0.1, 30)
            sample2 = np.random.normal(0.5, 0.1, 30)
            
            result = self.analyzer.welch_t_test(sample1.tolist(), sample2.tolist())
            if result.is_significant:
                false_positive_count += 1
                
        # Should be approximately 5% false positive rate
        false_positive_rate = false_positive_count / total_tests
        assert false_positive_rate <= 0.10  # Allow some variance
        
    def test_power_analysis_accuracy(self):
        """Test statistical power analysis accuracy."""
        # Generate samples with known effect size
        effect_size = 0.8  # Large effect
        np.random.seed(42)
        
        sample1 = np.random.normal(0.8, 0.1, 50)
        sample2 = np.random.normal(0.6, 0.1, 50)
        
        result = self.analyzer.welch_t_test(sample1.tolist(), sample2.tolist())
        
        # Should detect large effect size
        assert abs(result.effect_size) > 0.5
        assert result.is_significant
        assert result.power_analysis['observed_power'] > 0.8
        
    def test_confidence_interval_coverage(self):
        """Test confidence interval coverage probability."""
        true_difference = 0.2
        coverage_count = 0
        total_tests = 100
        
        np.random.seed(42)
        
        for _ in range(total_tests):
            sample1 = np.random.normal(0.7, 0.1, 30)
            sample2 = np.random.normal(0.5, 0.1, 30)  # True difference = 0.2
            
            result = self.analyzer.welch_t_test(sample1.tolist(), sample2.tolist())
            ci_lower, ci_upper = result.confidence_interval
            
            if ci_lower <= true_difference <= ci_upper:
                coverage_count += 1
                
        coverage_rate = coverage_count / total_tests
        # Should be approximately 95% coverage
        assert coverage_rate >= 0.90  # Allow some variance


if __name__ == "__main__":
    pytest.main([__file__, "-v"])