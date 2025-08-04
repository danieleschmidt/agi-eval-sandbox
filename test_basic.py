#!/usr/bin/env python3
"""
Basic functionality test without external dependencies.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test core imports."""
    try:
        from agi_eval_sandbox.core.benchmarks import TruthfulQABenchmark, Question, QuestionType, Score
        print("✅ Benchmarks module imported")
        
        from agi_eval_sandbox.core.models import LocalProvider, ModelConfig
        print("✅ Models module imported")
        
        from agi_eval_sandbox.core.results import Results, BenchmarkResult, EvaluationResult
        print("✅ Results module imported")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_benchmarks():
    """Test benchmark functionality."""
    try:
        from agi_eval_sandbox.core.benchmarks import TruthfulQABenchmark
        
        benchmark = TruthfulQABenchmark()
        assert benchmark.name == "truthfulqa"
        assert benchmark.version == "1.0"
        
        questions = benchmark.get_questions()
        assert len(questions) == 3
        print(f"✅ TruthfulQA benchmark has {len(questions)} questions")
        
        # Test evaluation
        q1 = questions[0]
        correct_response = "Nothing happens if you eat watermelon seeds."
        score = benchmark.evaluate_response(q1, correct_response)
        assert score.passed is True
        print("✅ Question evaluation working")
        
        return True
    except Exception as e:
        print(f"❌ Benchmark test failed: {e}")
        return False

def test_models():
    """Test model functionality."""
    try:
        from agi_eval_sandbox.core.models import LocalProvider, ModelConfig
        import asyncio
        
        config = ModelConfig(name="test-model", provider="local")
        provider = LocalProvider(config)
        
        # Test sync functionality
        limits = provider.get_limits()
        assert limits.requests_per_minute == 1000
        print("✅ Local provider limits working")
        
        # Test async functionality
        async def test_generation():
            response = await provider.generate("Test prompt")
            assert "Mock response for: Test prompt" in response
            return True
        
        result = asyncio.run(test_generation())
        assert result
        print("✅ Local provider generation working")
        
        return True
    except Exception as e:
        print(f"❌ Model test failed: {e}")
        return False

def test_results():
    """Test results functionality."""
    try:
        from agi_eval_sandbox.core.results import Results, BenchmarkResult, EvaluationResult
        from agi_eval_sandbox.core.benchmarks import Score
        
        results = Results()
        
        # Create mock evaluation result
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
        
        summary = results.summary()
        assert summary["total_benchmarks"] == 1
        assert summary["total_questions"] == 1
        assert summary["overall_score"] == 0.8
        print("✅ Results summary working")
        
        return True
    except Exception as e:
        print(f"❌ Results test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🧪 Running basic functionality tests...")
    print("=" * 50)
    
    tests = [
        ("Imports", test_imports),
        ("Benchmarks", test_benchmarks),
        ("Models", test_models),
        ("Results", test_results),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n📋 Testing {name}:")
        if test_func():
            passed += 1
            print(f"✅ {name} tests passed")
        else:
            print(f"❌ {name} tests failed")
    
    print("\n" + "=" * 50)
    print(f"🎯 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All tests passed! Generation 1 implementation is working.")
        return True
    else:
        print("💥 Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)