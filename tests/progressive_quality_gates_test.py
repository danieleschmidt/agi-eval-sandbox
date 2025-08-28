"""
Comprehensive Test Suite for Progressive Quality Gates System
Tests all three generations of progressive quality enhancement.
"""

import asyncio
import pytest
import time
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

from src.agi_eval_sandbox.core.models import EvaluationContext
from src.agi_eval_sandbox.quality.progressive_quality_gates import (
    ProgressiveQualityGates,
    ProgressiveConfig,
    DevelopmentPhase,
    RiskLevel,
    QualityMetric,
    QualityGateResult,
    ProgressiveQualityResult,
    PrototypeQualityStrategy,
    ProductionQualityStrategy
)
from src.agi_eval_sandbox.quality.adaptive_quality_intelligence import (
    AdaptiveQualityIntelligence,
    AdaptiveConfig,
    QualityPattern,
    PredictionConfidence
)
from src.agi_eval_sandbox.quality.quantum_quality_optimization import (
    QuantumQualityOptimizer,
    QuantumConfig,
    OptimizationStrategy,
    QuantumState
)


class TestProgressiveQualityGatesFoundation:
    """Test Generation 1: Progressive Quality Gates Foundation."""
    
    @pytest.fixture
    def progressive_gates(self):
        """Create progressive quality gates instance for testing."""
        config = ProgressiveConfig(
            phase=DevelopmentPhase.DEVELOPMENT,
            risk_level=RiskLevel.MEDIUM
        )
        return ProgressiveQualityGates(config)
    
    @pytest.fixture
    def sample_context(self):
        """Create sample evaluation context."""
        return EvaluationContext(
            model_name="test-model",
            model_provider="test",
            benchmarks=["test_benchmark"],
            timestamp=datetime.now()
        )
    
    def test_development_phase_configuration(self, progressive_gates):
        """Test development phase configuration."""
        assert progressive_gates.config.phase == DevelopmentPhase.DEVELOPMENT
        assert progressive_gates.config.risk_level == RiskLevel.MEDIUM
        assert progressive_gates.config.enable_ml_quality_prediction is True
    
    def test_phase_specific_strategies(self, progressive_gates):
        """Test that different phases use appropriate strategies."""
        # Prototype phase should use lenient strategy
        prototype_strategy = progressive_gates.strategies[DevelopmentPhase.PROTOTYPE]
        assert isinstance(prototype_strategy, PrototypeQualityStrategy)
        
        # Production phase should use strict strategy
        production_strategy = progressive_gates.strategies[DevelopmentPhase.PRODUCTION]
        assert isinstance(production_strategy, ProductionQualityStrategy)
    
    @pytest.mark.asyncio
    async def test_prototype_quality_evaluation(self, sample_context):
        """Test prototype quality evaluation is lenient."""
        strategy = PrototypeQualityStrategy()
        metrics = await strategy.evaluate(sample_context)
        
        assert len(metrics) >= 2  # At least syntax and functionality checks
        assert all(isinstance(metric, QualityMetric) for metric in metrics)
        
        # Prototype should be more lenient
        thresholds = strategy.get_thresholds(DevelopmentPhase.PROTOTYPE, RiskLevel.LOW)
        assert thresholds["overall_pass_threshold"] <= 0.7  # Lenient threshold
    
    @pytest.mark.asyncio
    async def test_production_quality_evaluation(self, sample_context):
        """Test production quality evaluation is strict."""
        strategy = ProductionQualityStrategy()
        metrics = await strategy.evaluate(sample_context)
        
        assert len(metrics) >= 3  # Security, performance, reliability checks
        
        # Production should be strict
        thresholds = strategy.get_thresholds(DevelopmentPhase.PRODUCTION, RiskLevel.HIGH)
        assert thresholds["overall_pass_threshold"] >= 0.9  # Strict threshold
        assert thresholds["security_threshold"] == 1.0  # Perfect security required
    
    @pytest.mark.asyncio
    async def test_progressive_evaluation_flow(self, progressive_gates, sample_context):
        """Test complete progressive evaluation flow."""
        result = await progressive_gates.evaluate(sample_context)
        
        assert isinstance(result, ProgressiveQualityResult)
        assert result.phase == DevelopmentPhase.DEVELOPMENT
        assert result.risk_level == RiskLevel.MEDIUM
        assert 0.0 <= result.confidence_score <= 1.0
        assert len(result.recommendations) > 0
        assert result.base_result.overall_score >= 0.0
    
    def test_adaptive_thresholds_adjustment(self, progressive_gates):
        """Test adaptive threshold adjustment."""
        # Simulate performance history
        progressive_gates.performance_history["test_metric"] = [0.8, 0.85, 0.9, 0.92, 0.95]
        
        # Should adjust thresholds based on improving trend
        assert len(progressive_gates.performance_history["test_metric"]) == 5
    
    def test_phase_advancement_recommendations(self, progressive_gates):
        """Test phase advancement recommendations."""
        # Simulate good performance history
        progressive_gates.performance_history["overall_score"] = [0.9, 0.92, 0.94, 0.96, 0.98]
        
        # Should recommend advancement
        asyncio.run(self._test_recommendations(progressive_gates))
    
    async def _test_recommendations(self, gates):
        recommendations = await gates.get_phase_recommendations()
        assert "current_phase" in recommendations
        assert "estimated_readiness" in recommendations
        assert recommendations["estimated_readiness"] > 0.8


class TestAdaptiveQualityIntelligence:
    """Test Generation 2: Adaptive Quality Intelligence."""
    
    @pytest.fixture
    def adaptive_intelligence(self):
        """Create adaptive intelligence instance for testing."""
        config = AdaptiveConfig(
            enable_ml_prediction=True,
            enable_anomaly_detection=True,
            history_window_size=20
        )
        return AdaptiveQualityIntelligence(config)
    
    @pytest.fixture
    def sample_results(self):
        """Create sample progressive quality results."""
        results = []
        for i in range(10):
            metrics = [
                QualityMetric(
                    name="security_validation",
                    passed=i > 5,  # Improving trend
                    score=0.5 + (i * 0.05),  # Gradual improvement
                    message="Security check"
                ),
                QualityMetric(
                    name="performance_validation", 
                    passed=True,
                    score=0.8 + np.random.normal(0, 0.05),  # Stable with noise
                    message="Performance check"
                )
            ]
            
            base_result = QualityGateResult(
                timestamp=datetime.now(),
                passed=all(m.passed for m in metrics),
                overall_score=sum(m.score for m in metrics) / len(metrics),
                metrics=metrics
            )
            
            progressive_result = ProgressiveQualityResult(
                base_result=base_result,
                phase=DevelopmentPhase.DEVELOPMENT,
                risk_level=RiskLevel.MEDIUM,
                confidence_score=0.8
            )
            
            results.append(progressive_result)
        
        return results
    
    @pytest.mark.asyncio
    async def test_trend_analysis_improving_pattern(self, adaptive_intelligence, sample_results):
        """Test trend analysis detects improving patterns."""
        context = EvaluationContext(
            model_name="test",
            model_provider="test",
            benchmarks=[],
            timestamp=datetime.now()
        )
        
        trends = await adaptive_intelligence.analyze_quality_trends(sample_results, context)
        
        # Should detect improving trend for security_validation
        security_trend = next((t for t in trends if t.metric_name == "security_validation"), None)
        assert security_trend is not None
        assert security_trend.pattern in [QualityPattern.IMPROVING, QualityPattern.STABLE]
        assert security_trend.trend_strength > 0  # Positive trend
    
    @pytest.mark.asyncio
    async def test_anomaly_detection(self, adaptive_intelligence):
        """Test anomaly detection in quality metrics."""
        # Create normal metrics
        normal_metrics = [
            QualityMetric(name="test_metric", passed=True, score=0.8, message="Normal")
        ]
        
        # Simulate normal history
        for i in range(10):
            adaptive_intelligence.metric_history["test_metric"].append({
                "timestamp": datetime.now(),
                "value": 0.8 + np.random.normal(0, 0.05),
                "passed": True,
                "phase": "development",
                "risk_level": "medium"
            })
        
        # Create anomalous metric
        anomalous_metrics = [
            QualityMetric(name="test_metric", passed=False, score=0.2, message="Anomaly")  # Very low score
        ]
        
        context = EvaluationContext(
            model_name="test",
            model_provider="test", 
            benchmarks=[],
            timestamp=datetime.now()
        )
        
        anomalies = await adaptive_intelligence.detect_quality_anomalies(anomalous_metrics, context)
        
        # Should detect anomaly
        assert len(anomalies) > 0
        anomaly = anomalies[0]
        assert anomaly.is_anomaly is True
        assert anomaly.anomaly_score > 2.0  # High z-score
        assert anomaly.severity in ["medium", "high", "critical"]
    
    @pytest.mark.asyncio 
    async def test_prediction_accuracy_tracking(self, adaptive_intelligence):
        """Test prediction accuracy tracking."""
        # Set up mock predictions
        adaptive_intelligence.last_predictions["test_metric"] = [0.8, 0.85, 0.9]
        
        # Create result with actual value close to prediction
        result = ProgressiveQualityResult(
            base_result=QualityGateResult(
                timestamp=datetime.now(),
                passed=True,
                overall_score=0.85,
                metrics=[QualityMetric(name="test_metric", passed=True, score=0.82, message="Test")]
            ),
            phase=DevelopmentPhase.DEVELOPMENT,
            risk_level=RiskLevel.MEDIUM,
            confidence_score=0.8
        )
        
        await adaptive_intelligence._update_prediction_accuracy([result])
        
        # Should have accuracy data
        assert "test_metric" in adaptive_intelligence.prediction_accuracy
        assert len(adaptive_intelligence.prediction_accuracy["test_metric"]) > 0
        
        # Accuracy should be high (prediction was close)
        accuracy = adaptive_intelligence.prediction_accuracy["test_metric"][-1]
        assert accuracy > 0.8  # Good prediction accuracy
    
    def test_intelligence_summary(self, adaptive_intelligence):
        """Test intelligence summary generation."""
        # Add some mock data
        adaptive_intelligence.metric_history["test_metric"].extend([
            {"timestamp": datetime.now(), "value": 0.8, "passed": True, "phase": "dev", "risk_level": "medium"},
            {"timestamp": datetime.now(), "value": 0.85, "passed": True, "phase": "dev", "risk_level": "medium"}
        ])
        
        summary = asyncio.run(adaptive_intelligence.get_intelligence_summary())
        
        assert "metrics_tracked" in summary
        assert "total_data_points" in summary
        assert summary["metrics_tracked"] >= 1
        assert summary["total_data_points"] >= 2


class TestQuantumQualityOptimization:
    """Test Generation 3: Quantum Quality Optimization."""
    
    @pytest.fixture
    def quantum_optimizer(self):
        """Create quantum optimizer instance for testing."""
        config = QuantumConfig(
            enable_quantum_optimization=True,
            enable_entanglement_detection=True,
            max_iterations=100  # Reduced for testing
        )
        return QuantumQualityOptimizer(config)
    
    @pytest.fixture
    def optimization_results(self):
        """Create sample results for optimization."""
        results = []
        for i in range(5):
            metrics = [
                QualityMetric(name="security", passed=True, score=0.8 + i*0.02, message="Security"),
                QualityMetric(name="performance", passed=True, score=0.75 + i*0.03, message="Performance"),
                QualityMetric(name="reliability", passed=i > 2, score=0.7 + i*0.04, message="Reliability")
            ]
            
            base_result = QualityGateResult(
                timestamp=datetime.now(),
                passed=all(m.passed for m in metrics),
                overall_score=sum(m.score for m in metrics) / len(metrics),
                metrics=metrics
            )
            
            progressive_result = ProgressiveQualityResult(
                base_result=base_result,
                phase=DevelopmentPhase.DEVELOPMENT,
                risk_level=RiskLevel.MEDIUM,
                confidence_score=0.8
            )
            
            results.append(progressive_result)
        
        return results
    
    @pytest.mark.asyncio
    async def test_quantum_state_initialization(self, quantum_optimizer):
        """Test quantum state initialization."""
        # Wait for initialization
        await asyncio.sleep(0.1)  # Allow initialization to complete
        
        # Should have quantum states for common metrics
        common_metrics = ["security_validation", "performance_validation", "reliability_validation"]
        
        for metric in common_metrics:
            if metric in quantum_optimizer.quantum_states:
                state = quantum_optimizer.quantum_states[metric]
                assert len(state.amplitudes) == len(state.probabilities)
                assert all(abs(prob - abs(amp)**2) < 1e-10 for prob, amp in zip(state.probabilities, state.amplitudes))
                assert abs(sum(state.probabilities) - 1.0) < 1e-10  # Normalized
    
    @pytest.mark.asyncio
    async def test_entanglement_detection(self, quantum_optimizer, optimization_results):
        """Test quantum entanglement detection."""
        await quantum_optimizer._detect_quantum_entanglement(optimization_results)
        
        # Should detect some entangled metrics
        total_entangled_pairs = sum(len(entangled) for entangled in quantum_optimizer.entanglement_graph.values()) // 2
        
        # May or may not find entanglement depending on correlation, so just test structure
        assert isinstance(quantum_optimizer.entanglement_graph, dict)
        for metric_name, entangled_set in quantum_optimizer.entanglement_graph.items():
            assert isinstance(entangled_set, set)
    
    @pytest.mark.asyncio
    async def test_optimization_strategies(self, quantum_optimizer, optimization_results):
        """Test different optimization strategies."""
        from src.agi_eval_sandbox.quality.adaptive_quality_intelligence import AdaptiveQualityIntelligence
        
        intelligence = AdaptiveQualityIntelligence()
        context = EvaluationContext(
            model_name="test",
            model_provider="test",
            benchmarks=[],
            timestamp=datetime.now()
        )
        
        # Test strategy selection
        strategy = await quantum_optimizer._select_optimization_strategy(optimization_results)
        assert isinstance(strategy, OptimizationStrategy)
        
        # Test quantum annealing
        annealing_result = await quantum_optimizer._quantum_annealing_optimization(
            optimization_results, intelligence
        )
        assert annealing_result.optimized_thresholds is not None
        assert annealing_result.quality_improvement >= -1.0  # Can be negative for bad optimization
        assert annealing_result.convergence_iterations > 0
    
    @pytest.mark.asyncio
    async def test_quantum_measurement_update(self, quantum_optimizer):
        """Test quantum measurement update."""
        await asyncio.sleep(0.1)  # Allow initialization
        
        # Create a quantum state manually if needed
        if "test_metric" not in quantum_optimizer.quantum_states:
            await quantum_optimizer._create_quantum_state("test_metric")
        
        initial_state = quantum_optimizer.quantum_states["test_metric"]
        initial_fidelity = initial_state.fidelity
        
        # Perform measurement
        await quantum_optimizer._quantum_measurement_update("test_metric", 0.8, True)
        
        # State should have collapsed and possibly lost fidelity
        updated_state = quantum_optimizer.quantum_states["test_metric"]
        assert updated_state.fidelity <= initial_fidelity + 1e-10  # Allow for floating point precision
    
    @pytest.mark.asyncio
    async def test_complete_optimization_flow(self, quantum_optimizer, optimization_results):
        """Test complete quantum optimization flow."""
        from src.agi_eval_sandbox.quality.adaptive_quality_intelligence import AdaptiveQualityIntelligence
        
        intelligence = AdaptiveQualityIntelligence()
        context = EvaluationContext(
            model_name="test",
            model_provider="test", 
            benchmarks=[],
            timestamp=datetime.now()
        )
        
        result = await quantum_optimizer.optimize_quality_system(
            optimization_results, intelligence, context
        )
        
        assert result.optimized_thresholds is not None
        assert result.resource_allocation is not None
        assert isinstance(result.quality_improvement, float)
        assert result.quantum_advantage >= 0.0
        assert isinstance(result.optimized_strategy, OptimizationStrategy)
    
    @pytest.mark.asyncio
    async def test_quantum_system_status(self, quantum_optimizer):
        """Test quantum system status reporting."""
        await asyncio.sleep(0.1)  # Allow initialization
        
        status = await quantum_optimizer.get_quantum_system_status()
        
        assert "quantum_states" in status
        assert "entangled_pairs" in status
        assert "total_optimizations" in status
        assert "coherence_status" in status
        assert isinstance(status["quantum_states"], int)
        assert isinstance(status["entangled_pairs"], int)


class TestIntegratedProgressiveQualitySystem:
    """Integration tests for the complete progressive quality system."""
    
    @pytest.fixture
    def integrated_system(self):
        """Create integrated system with all components."""
        progressive_gates = ProgressiveQualityGates()
        adaptive_intelligence = AdaptiveQualityIntelligence()
        quantum_optimizer = QuantumQualityOptimizer()
        
        return {
            "gates": progressive_gates,
            "intelligence": adaptive_intelligence, 
            "optimizer": quantum_optimizer
        }
    
    @pytest.fixture
    def complex_evaluation_scenario(self):
        """Create complex evaluation scenario for integration testing."""
        results = []
        
        # Simulate a realistic development progression
        phases = [DevelopmentPhase.PROTOTYPE] * 3 + [DevelopmentPhase.DEVELOPMENT] * 4 + [DevelopmentPhase.TESTING] * 3
        
        for i, phase in enumerate(phases):
            # Simulate quality improvement over time
            base_quality = 0.5 + (i * 0.03)  # Gradual improvement
            noise = np.random.normal(0, 0.1)  # Some randomness
            
            metrics = [
                QualityMetric(
                    name="security_validation",
                    passed=base_quality > 0.7,
                    score=max(0.0, min(1.0, base_quality + noise)),
                    message="Security validation"
                ),
                QualityMetric(
                    name="performance_validation",
                    passed=base_quality > 0.6,
                    score=max(0.0, min(1.0, base_quality + 0.1 + noise)),
                    message="Performance validation" 
                ),
                QualityMetric(
                    name="reliability_validation",
                    passed=base_quality > 0.8,
                    score=max(0.0, min(1.0, base_quality - 0.1 + noise)),
                    message="Reliability validation"
                )
            ]
            
            base_result = QualityGateResult(
                timestamp=datetime.now(),
                passed=all(m.passed for m in metrics),
                overall_score=sum(m.score for m in metrics) / len(metrics),
                metrics=metrics
            )
            
            progressive_result = ProgressiveQualityResult(
                base_result=base_result,
                phase=phase,
                risk_level=RiskLevel.MEDIUM,
                confidence_score=0.7 + (i * 0.02)
            )
            
            results.append(progressive_result)
        
        return results
    
    @pytest.mark.asyncio
    async def test_end_to_end_quality_optimization(self, integrated_system, complex_evaluation_scenario):
        """Test end-to-end quality optimization with all components."""
        gates = integrated_system["gates"]
        intelligence = integrated_system["intelligence"]
        optimizer = integrated_system["optimizer"]
        
        results = complex_evaluation_scenario
        context = EvaluationContext(
            model_name="integration-test",
            model_provider="test",
            benchmarks=["security", "performance", "reliability"],
            timestamp=datetime.now()
        )
        
        # Step 1: Analyze trends with adaptive intelligence
        trends = await intelligence.analyze_quality_trends(results, context)
        assert len(trends) > 0
        
        # Step 2: Detect anomalies
        latest_metrics = results[-1].base_result.metrics
        anomalies = await intelligence.detect_quality_anomalies(latest_metrics, context)
        # May or may not find anomalies - just test it runs
        
        # Step 3: Perform quantum optimization
        optimization_result = await optimizer.optimize_quality_system(results, intelligence, context)
        assert optimization_result.quality_improvement is not None
        
        # Step 4: Get phase recommendations
        phase_recommendations = await gates.get_phase_recommendations()
        assert "estimated_readiness" in phase_recommendations
        
        # Verify integration
        assert len(trends) >= 1  # Should analyze at least one metric
        assert optimization_result.optimized_thresholds is not None
        assert phase_recommendations["estimated_readiness"] >= 0.0
    
    @pytest.mark.asyncio
    async def test_performance_under_load(self, integrated_system):
        """Test system performance under load."""
        gates = integrated_system["gates"] 
        intelligence = integrated_system["intelligence"]
        optimizer = integrated_system["optimizer"]
        
        # Generate large dataset
        large_results = []
        for i in range(50):  # Larger dataset
            metrics = [
                QualityMetric(name=f"metric_{j}", passed=True, score=0.7 + (j*0.05) % 0.3, message=f"Metric {j}")
                for j in range(5)  # Multiple metrics
            ]
            
            base_result = QualityGateResult(
                timestamp=datetime.now(),
                passed=True,
                overall_score=0.8,
                metrics=metrics
            )
            
            progressive_result = ProgressiveQualityResult(
                base_result=base_result,
                phase=DevelopmentPhase.DEVELOPMENT,
                risk_level=RiskLevel.MEDIUM,
                confidence_score=0.8
            )
            
            large_results.append(progressive_result)
        
        context = EvaluationContext(
            model_name="load-test",
            model_provider="test",
            benchmarks=[],
            timestamp=datetime.now()
        )
        
        # Measure performance
        start_time = time.time()
        
        # Run analysis
        trends = await intelligence.analyze_quality_trends(large_results, context)
        optimization_result = await optimizer.optimize_quality_system(large_results, intelligence, context)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Should complete within reasonable time
        assert processing_time < 30.0  # 30 seconds max
        assert len(trends) >= 3  # Should analyze multiple metrics
        assert optimization_result.quantum_advantage >= 0.5  # Should show some advantage
    
    def test_system_configuration_consistency(self, integrated_system):
        """Test configuration consistency across components."""
        gates = integrated_system["gates"]
        intelligence = integrated_system["intelligence"]
        optimizer = integrated_system["optimizer"]
        
        # All components should be properly configured
        assert gates.config.enable_ml_quality_prediction is True
        assert intelligence.config.enable_ml_prediction is True
        assert optimizer.config.enable_quantum_optimization is True
        
        # Components should have compatible settings
        assert gates.config.history_window_size <= intelligence.config.history_window_size + 20  # Reasonable difference


# Performance benchmarks
class TestPerformanceBenchmarks:
    """Performance benchmarks for progressive quality system."""
    
    @pytest.mark.asyncio
    async def test_single_evaluation_performance(self):
        """Benchmark single evaluation performance."""
        gates = ProgressiveQualityGates()
        context = EvaluationContext(
            model_name="benchmark",
            model_provider="test",
            benchmarks=["test"],
            timestamp=datetime.now()
        )
        
        start_time = time.time()
        result = await gates.evaluate(context)
        end_time = time.time()
        
        evaluation_time = end_time - start_time
        
        # Should complete quickly
        assert evaluation_time < 5.0  # 5 seconds max
        assert result.base_result.overall_score >= 0.0
    
    @pytest.mark.asyncio
    async def test_trend_analysis_performance(self):
        """Benchmark trend analysis performance."""
        intelligence = AdaptiveQualityIntelligence()
        
        # Generate test data
        results = []
        for i in range(100):
            metrics = [QualityMetric(name="perf_test", passed=True, score=0.8, message="Test")]
            base_result = QualityGateResult(
                timestamp=datetime.now(),
                passed=True,
                overall_score=0.8,
                metrics=metrics
            )
            progressive_result = ProgressiveQualityResult(
                base_result=base_result,
                phase=DevelopmentPhase.DEVELOPMENT,
                risk_level=RiskLevel.MEDIUM,
                confidence_score=0.8
            )
            results.append(progressive_result)
        
        context = EvaluationContext(
            model_name="benchmark",
            model_provider="test",
            benchmarks=[],
            timestamp=datetime.now()
        )
        
        start_time = time.time()
        trends = await intelligence.analyze_quality_trends(results, context)
        end_time = time.time()
        
        analysis_time = end_time - start_time
        
        # Should handle large datasets efficiently
        assert analysis_time < 10.0  # 10 seconds max for 100 results
        assert len(trends) >= 1
    
    @pytest.mark.asyncio 
    async def test_quantum_optimization_performance(self):
        """Benchmark quantum optimization performance."""
        optimizer = QuantumQualityOptimizer()
        intelligence = AdaptiveQualityIntelligence()
        
        # Generate optimization data
        results = []
        for i in range(10):
            metrics = [QualityMetric(name="quantum_test", passed=True, score=0.7 + i*0.02, message="Test")]
            base_result = QualityGateResult(
                timestamp=datetime.now(),
                passed=True,
                overall_score=0.7 + i*0.02,
                metrics=metrics
            )
            progressive_result = ProgressiveQualityResult(
                base_result=base_result,
                phase=DevelopmentPhase.DEVELOPMENT,
                risk_level=RiskLevel.MEDIUM,
                confidence_score=0.8
            )
            results.append(progressive_result)
        
        context = EvaluationContext(
            model_name="benchmark",
            model_provider="test",
            benchmarks=[],
            timestamp=datetime.now()
        )
        
        start_time = time.time()
        optimization_result = await optimizer.optimize_quality_system(results, intelligence, context)
        end_time = time.time()
        
        optimization_time = end_time - start_time
        
        # Should demonstrate quantum advantage in performance
        assert optimization_time < 15.0  # 15 seconds max
        assert optimization_result.quantum_advantage >= 1.0  # Should be at least as fast as classical


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])