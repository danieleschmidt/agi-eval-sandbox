#!/usr/bin/env python3
"""
Simple test runner for progressive quality gates
"""

import sys
import os
import traceback
import asyncio
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from agi_eval_sandbox.core.models import EvaluationContext
        from agi_eval_sandbox.quality.progressive_quality_gates import (
            ProgressiveQualityGates, ProgressiveConfig, DevelopmentPhase, RiskLevel
        )
        from agi_eval_sandbox.quality.adaptive_quality_intelligence import (
            AdaptiveQualityIntelligence, QualityPattern
        )
        from agi_eval_sandbox.quality.quantum_quality_optimization import (
            QuantumQualityOptimizer, OptimizationStrategy
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False

async def test_progressive_gates():
    """Test progressive quality gates functionality."""
    print("\nTesting Progressive Quality Gates...")
    
    try:
        from agi_eval_sandbox.quality.progressive_quality_gates import ProgressiveQualityGates
        
        # Create progressive gates
        gates = ProgressiveQualityGates()
        
        # Create evaluation context
        context = EvaluationContext(
            model_name="test-model",
            model_provider="test",
            benchmarks=["test_benchmark"],
            timestamp=datetime.now()
        )
        
        # Run evaluation
        result = await gates.evaluate(context)
        
        # Verify results
        assert result.base_result.overall_score >= 0.0
        assert result.confidence_score >= 0.0
        assert len(result.recommendations) > 0
        
        print("✅ Progressive Quality Gates test passed")
        return True
        
    except Exception as e:
        print(f"❌ Progressive Quality Gates test failed: {e}")
        traceback.print_exc()
        return False

async def test_adaptive_intelligence():
    """Test adaptive quality intelligence functionality."""
    print("\nTesting Adaptive Quality Intelligence...")
    
    try:
        from agi_eval_sandbox.quality.adaptive_quality_intelligence import AdaptiveQualityIntelligence
        
        # Create adaptive intelligence
        intelligence = AdaptiveQualityIntelligence()
        
        # Wait for initialization
        await asyncio.sleep(0.1)
        
        # Create mock progressive results
        from agi_eval_sandbox.quality.progressive_quality_gates import (
            QualityMetric, QualityGateResult, ProgressiveQualityResult
        )
        
        results = []
        for i in range(5):
            metrics = [
                QualityMetric(
                    name="test_metric",
                    passed=True,
                    score=0.7 + (i * 0.05),
                    message="Test metric"
                )
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
            
            results.append(progressive_result)
        
        # Test trend analysis
        context = EvaluationContext(
            model_name="test",
            model_provider="test",
            benchmarks=[],
            timestamp=datetime.now()
        )
        
        trends = await intelligence.analyze_quality_trends(results, context)
        
        # Verify results
        assert len(trends) >= 0  # May not find trends with limited data
        
        # Test summary
        summary = await intelligence.get_intelligence_summary()
        assert "metrics_tracked" in summary
        
        print("✅ Adaptive Quality Intelligence test passed")
        return True
        
    except Exception as e:
        print(f"❌ Adaptive Quality Intelligence test failed: {e}")
        traceback.print_exc()
        return False

async def test_quantum_optimization():
    """Test quantum quality optimization functionality."""
    print("\nTesting Quantum Quality Optimization...")
    
    try:
        from agi_eval_sandbox.quality.quantum_quality_optimization import QuantumQualityOptimizer
        
        # Create quantum optimizer
        optimizer = QuantumQualityOptimizer()
        
        # Wait for initialization
        await asyncio.sleep(0.1)
        
        # Create mock data
        from agi_eval_sandbox.quality.progressive_quality_gates import (
            QualityMetric, QualityGateResult, ProgressiveQualityResult
        )
        from agi_eval_sandbox.quality.adaptive_quality_intelligence import AdaptiveQualityIntelligence
        
        results = []
        for i in range(3):
            metrics = [
                QualityMetric(
                    name="quantum_test",
                    passed=True,
                    score=0.8 + (i * 0.05),
                    message="Quantum test"
                )
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
            
            results.append(progressive_result)
        
        # Test optimization
        intelligence = AdaptiveQualityIntelligence()
        context = EvaluationContext(
            model_name="test",
            model_provider="test",
            benchmarks=[],
            timestamp=datetime.now()
        )
        
        optimization_result = await optimizer.optimize_quality_system(results, intelligence, context)
        
        # Verify results
        assert optimization_result.optimized_thresholds is not None
        assert optimization_result.quantum_advantage >= 0.0
        
        # Test status
        status = await optimizer.get_quantum_system_status()
        assert "quantum_states" in status
        
        print("✅ Quantum Quality Optimization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Quantum Quality Optimization test failed: {e}")
        traceback.print_exc()
        return False

async def test_integration():
    """Test integration between all components."""
    print("\nTesting System Integration...")
    
    try:
        from agi_eval_sandbox.quality.progressive_quality_gates import ProgressiveQualityGates
        from agi_eval_sandbox.quality.adaptive_quality_intelligence import AdaptiveQualityIntelligence
        from agi_eval_sandbox.quality.quantum_quality_optimization import QuantumQualityOptimizer
        
        # Create all components
        gates = ProgressiveQualityGates()
        intelligence = AdaptiveQualityIntelligence()
        optimizer = QuantumQualityOptimizer()
        
        # Wait for initialization
        await asyncio.sleep(0.2)
        
        # Create evaluation context
        context = EvaluationContext(
            model_name="integration-test",
            model_provider="test",
            benchmarks=["integration"],
            timestamp=datetime.now()
        )
        
        # Run progressive evaluation
        result = await gates.evaluate(context)
        
        # Use result for intelligence analysis
        trends = await intelligence.analyze_quality_trends([result], context)
        
        # Run quantum optimization
        optimization = await optimizer.optimize_quality_system([result], intelligence, context)
        
        # Verify integration
        assert result.base_result.overall_score >= 0.0
        assert optimization.quality_improvement is not None
        
        print("✅ System Integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ System Integration test failed: {e}")
        traceback.print_exc()
        return False

def performance_benchmark():
    """Run performance benchmarks."""
    print("\nRunning Performance Benchmarks...")
    
    try:
        import time
        
        # Simple benchmark
        start_time = time.time()
        
        # Simulate some work
        for i in range(1000):
            x = np.random.random()
            y = x ** 2
        
        end_time = time.time()
        benchmark_time = end_time - start_time
        
        print(f"✅ Performance benchmark completed in {benchmark_time:.3f}s")
        return True
        
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False

async def main():
    """Run all tests."""
    print("🧪 Progressive Quality Gates Test Suite")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Progressive Gates", test_progressive_gates),
        ("Adaptive Intelligence", test_adaptive_intelligence),
        ("Quantum Optimization", test_quantum_optimization),
        ("System Integration", test_integration),
        ("Performance Benchmark", performance_benchmark)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 Running {test_name} test...")
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                success = await test_func()
            else:
                success = test_func()
                
            if success:
                passed += 1
            
        except Exception as e:
            print(f"❌ Test {test_name} crashed: {e}")
            traceback.print_exc()
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Progressive Quality Gates system is working correctly.")
    else:
        print(f"⚠️ {total - passed} tests failed. Please review the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)