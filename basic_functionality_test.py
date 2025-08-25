#!/usr/bin/env python3
"""
Basic functionality test for AGI Evaluation Sandbox
Generation 1: Make It Work (Simple)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_basic_imports():
    """Test that core modules can be imported."""
    try:
        import agi_eval_sandbox
        from agi_eval_sandbox.core.evaluator import EvalSuite
        from agi_eval_sandbox.core.models import Model
        from agi_eval_sandbox.core.benchmarks import Benchmark
        print("✅ All core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_instantiation():
    """Test basic object creation."""
    try:
        from agi_eval_sandbox.core.evaluator import EvalSuite
        eval_suite = EvalSuite()
        print(f"✅ EvalSuite created with {len(eval_suite.list_benchmarks())} benchmarks")
        return True
    except Exception as e:
        print(f"❌ Instantiation failed: {e}")
        return False

def test_benchmark_listing():
    """Test benchmark listing functionality."""
    try:
        from agi_eval_sandbox.core.evaluator import EvalSuite
        eval_suite = EvalSuite()
        benchmarks = eval_suite.list_benchmarks()
        print(f"✅ Available benchmarks: {benchmarks}")
        return len(benchmarks) > 0
    except Exception as e:
        print(f"❌ Benchmark listing failed: {e}")
        return False

def main():
    """Run basic functionality tests."""
    print("🧪 Running Generation 1 Basic Functionality Tests...")
    print("=" * 50)
    
    tests = [
        test_basic_imports,
        test_basic_instantiation,
        test_benchmark_listing
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print(f"\nRunning {test.__name__}...")
        if test():
            passed += 1
        else:
            print(f"⚠️  Test {test.__name__} failed")
    
    print(f"\n{'=' * 50}")
    print(f"📊 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 Generation 1 (Make It Work) - ALL TESTS PASSED!")
        return True
    else:
        print("💥 Some tests failed. Basic functionality needs attention.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)