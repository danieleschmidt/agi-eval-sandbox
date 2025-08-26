#!/usr/bin/env python3
"""
Generation 1: MAKE IT WORK - Simple functionality verification
"""

import sys
sys.path.insert(0, '/root/repo/src')

def test_basic_imports():
    """Test that core modules can be imported"""
    try:
        from agi_eval_sandbox import EvalSuite, Model
        from agi_eval_sandbox.core.benchmarks import Benchmark
        from agi_eval_sandbox.core.results import Results
        print("✅ All core imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_model_creation():
    """Test basic model instantiation"""
    try:
        from agi_eval_sandbox.core.models import Model
        # Create a local model (valid provider)
        model = Model(provider="local", name="test-model")
        print(f"✅ Model created: {model.name}")
        return True
    except Exception as e:
        print(f"❌ Model creation error: {e}")
        return False

def test_eval_suite_creation():
    """Test EvalSuite instantiation"""
    try:
        from agi_eval_sandbox.core.evaluator import EvalSuite
        suite = EvalSuite()
        print("✅ EvalSuite created successfully")
        return True
    except Exception as e:
        print(f"❌ EvalSuite creation error: {e}")
        return False

def test_basic_api_structure():
    """Test API endpoints structure exists"""
    try:
        # Test the simple API that has fewer dependencies
        from agi_eval_sandbox.api.simple_main import app
        print("✅ Simple FastAPI app accessible")
        return True
    except Exception as e:
        print(f"❌ Simple API structure error: {e}")
        # Fallback to testing if main API module can be imported at all
        try:
            import agi_eval_sandbox.api.main
            print("✅ Main API module structure exists")
            return True  
        except Exception as e2:
            print(f"❌ API structure error: {e2}")
            return False

def main():
    """Run Generation 1 basic functionality tests"""
    print("🚀 GENERATION 1: MAKE IT WORK - Basic Functionality Test")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_basic_imports),
        ("Model Creation", test_model_creation),
        ("EvalSuite Creation", test_eval_suite_creation),
        ("API Structure", test_basic_api_structure),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n📋 Testing {name}...")
        if test_func():
            passed += 1
        
    print("\n" + "=" * 60)
    print(f"📊 RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 GENERATION 1 COMPLETE - Basic functionality working!")
        return True
    else:
        print("⚠️  Some basic functionality issues found")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)