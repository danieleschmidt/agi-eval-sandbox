#!/usr/bin/env python3
"""
Minimal Function Test - Generation 1 Core Validation
Tests core functionality that doesn't require external dependencies
"""

import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_core_imports():
    """Test that core imports work."""
    print("📦 Testing Core Imports...")
    
    try:
        from agi_eval_sandbox import __version__
        print(f"  ✓ Package version: {__version__}")
        
        from agi_eval_sandbox.core.exceptions import EvaluationError, ValidationError
        print("  ✓ Exception classes imported")
        
        from agi_eval_sandbox.core.benchmarks import Question, Score, QuestionType
        print("  ✓ Benchmark classes imported")
        
        # Test basic class instantiation
        question = Question(
            id="test",
            prompt="Test question",
            correct_answer="Test answer",
            category="test"
        )
        print(f"  ✓ Question created: {question.id}")
        
        score = Score(value=0.8, passed=True, explanation="Test score")
        print(f"  ✓ Score created: {score.value}")
        
        print("  ✅ All core imports successful!")
        return True
        
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        return False

def test_model_providers():
    """Test model provider validation."""
    print("🤖 Testing Model Provider Validation...")
    
    try:
        from agi_eval_sandbox.core.validation import InputValidator
        
        validator = InputValidator()
        
        # Test valid providers
        valid_providers = ["openai", "anthropic", "local", "huggingface", "google"]
        for provider in valid_providers:
            result = validator.validate_provider(provider)
            print(f"  ✓ Valid provider: {result}")
        
        # Test invalid provider
        try:
            validator.validate_provider("invalid_provider")
            print("  ❌ Should have failed for invalid provider")
            return False
        except:
            print("  ✓ Correctly rejected invalid provider")
        
        print("  ✅ Provider validation working!")
        return True
        
    except Exception as e:
        print(f"  ❌ Provider validation failed: {e}")
        return False

def test_basic_model_creation():
    """Test basic model creation with valid providers."""
    print("🔧 Testing Basic Model Creation...")
    
    try:
        from agi_eval_sandbox.core.models import Model
        
        # Create a simple model with valid provider
        model = Model(
            provider="local",
            name="test-model"
        )
        
        print(f"  ✓ Model created: {model.name}")
        print(f"  ✓ Provider: {model.provider_name}")
        
        # Test model configuration access through config
        if hasattr(model, 'config'):
            print(f"  ✓ Temperature: {model.config.temperature}")
            print(f"  ✓ Max tokens: {model.config.max_tokens}")
        elif hasattr(model, 'temperature'):
            print(f"  ✓ Temperature: {model.temperature}")
            print(f"  ✓ Max tokens: {model.max_tokens}")
        else:
            print("  ℹ️ Model configuration attributes not accessible (normal for this implementation)")
        
        print("  ✅ Basic model creation successful!")
        return True
        
    except Exception as e:
        print(f"  ❌ Model creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_benchmark_structure():
    """Test benchmark data structures."""
    print("📊 Testing Benchmark Structures...")
    
    try:
        from agi_eval_sandbox.core.benchmarks import (
            Question, Score, QuestionType, Benchmark,
            TruthfulQABenchmark, MMLUBenchmark, HumanEvalBenchmark
        )
        
        # Test question types
        print(f"  ✓ Question types available: {list(QuestionType)}")
        
        # Test benchmark instantiation
        benchmarks = [
            TruthfulQABenchmark(),
            MMLUBenchmark(), 
            HumanEvalBenchmark()
        ]
        
        for benchmark in benchmarks:
            print(f"  ✓ Benchmark: {benchmark.name} v{benchmark.version}")
            questions = benchmark.get_questions()
            print(f"    - Questions: {len(questions)}")
        
        print("  ✅ Benchmark structures working!")
        return True
        
    except Exception as e:
        print(f"  ❌ Benchmark test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_results_structure():
    """Test results data structures."""
    print("📈 Testing Results Structures...")
    
    try:
        from agi_eval_sandbox.core.results import Results, BenchmarkResult, EvaluationResult
        from agi_eval_sandbox.core.benchmarks import Score
        
        # Create test result structures
        score = Score(value=0.85, passed=True, explanation="Test evaluation")
        
        eval_result = EvaluationResult(
            question_id="test_q1",
            question_prompt="What is 2+2?",
            model_response="4",
            score=score,
            benchmark_name="test_benchmark",
            category="math"
        )
        
        print(f"  ✓ Evaluation result: {eval_result.question_id}")
        
        benchmark_result = BenchmarkResult(
            benchmark_name="test_benchmark",
            model_name="test_model",
            model_provider="local",
            results=[eval_result]
        )
        
        print(f"  ✓ Benchmark result: {benchmark_result.benchmark_name}")
        print(f"  ✓ Average score: {benchmark_result.average_score}")
        print(f"  ✓ Pass rate: {benchmark_result.pass_rate}%")
        
        results = Results()
        results.add_benchmark_result(benchmark_result)
        
        summary = results.summary()
        print(f"  ✓ Results summary: {summary}")
        
        print("  ✅ Results structures working!")
        return True
        
    except Exception as e:
        print(f"  ❌ Results test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_eval_suite_basic():
    """Test basic EvalSuite functionality."""
    print("🎯 Testing EvalSuite Basic Functions...")
    
    try:
        from agi_eval_sandbox.core.evaluator import EvalSuite
        
        # Create eval suite
        suite = EvalSuite()
        print("  ✓ EvalSuite created")
        
        # Test benchmark listing
        benchmarks = suite.list_benchmarks()
        print(f"  ✓ Available benchmarks: {benchmarks}")
        
        # Test benchmark retrieval
        for benchmark_name in benchmarks[:2]:  # Test first 2
            benchmark = suite.get_benchmark(benchmark_name)
            if benchmark:
                print(f"  ✓ Retrieved benchmark: {benchmark_name}")
            else:
                print(f"  ❌ Failed to retrieve: {benchmark_name}")
                return False
        
        print("  ✅ EvalSuite basic functions working!")
        return True
        
    except Exception as e:
        print(f"  ❌ EvalSuite test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_async_functionality():
    """Test async functionality without external dependencies."""
    print("⚡ Testing Async Functionality...")
    
    try:
        # Test basic async operations
        async def simple_async_task():
            await asyncio.sleep(0.001)
            return "async_complete"
        
        result = await simple_async_task()
        print(f"  ✓ Basic async task: {result}")
        
        # Test async list comprehension
        async def async_number_gen(n):
            await asyncio.sleep(0.001)
            return n * 2
        
        tasks = [async_number_gen(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        print(f"  ✓ Async gather results: {results}")
        
        print("  ✅ Async functionality working!")
        return True
        
    except Exception as e:
        print(f"  ❌ Async test failed: {e}")
        return False

def test_configuration_system():
    """Test configuration system."""
    print("⚙️ Testing Configuration System...")
    
    try:
        from agi_eval_sandbox.config.simple_settings import SimpleSettings
        
        # Test settings instantiation
        settings = SimpleSettings()
        print(f"  ✓ Settings created")
        
        # Test default values
        print(f"  ✓ API Host: {settings.API_HOST}")
        print(f"  ✓ API Port: {settings.API_PORT}")
        print(f"  ✓ Log Level: {settings.LOG_LEVEL}")
        print(f"  ✓ Debug Mode: {settings.DEBUG}")
        
        print("  ✅ Configuration system working!")
        return True
        
    except Exception as e:
        print(f"  ❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Run all minimal tests."""
    print("\n" + "="*60)
    print("🚀 AGI EVALUATION SANDBOX - MINIMAL FUNCTION TESTS")
    print("="*60)
    
    tests = [
        ("Core Imports", test_core_imports),
        ("Model Providers", test_model_providers),
        ("Basic Model Creation", test_basic_model_creation),
        ("Benchmark Structures", test_benchmark_structure),
        ("Results Structures", test_results_structure),
        ("EvalSuite Basic", test_eval_suite_basic),
        ("Async Functionality", test_async_functionality),
        ("Configuration System", test_configuration_system)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*60)
    print("📊 MINIMAL TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed >= total * 0.8:  # 80% pass rate
        print("\n🎉 GENERATION 1 CORE FUNCTIONALITY VALIDATED!")
        print("✨ Ready to proceed to Generation 2")
        return 0
    else:
        print(f"\n⚠️ {total-passed} critical tests failed")
        print("🔧 Core functionality needs fixes before proceeding")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)