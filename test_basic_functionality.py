#!/usr/bin/env python3
"""
Basic functionality tests that don't require heavy dependencies.
Tests core functionality without ML libraries.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_import():
    """Test that core modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from agi_eval_sandbox.core.exceptions import ValidationError, ConfigurationError
        print("✅ Core exceptions imported")
        
        from agi_eval_sandbox.config.settings import Settings
        print("✅ Settings imported") 
        
        print("✅ All basic imports successful")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False


def test_cli_import():
    """Test that CLI can be imported."""
    print("\n🧪 Testing CLI import...")
    
    try:
        from agi_eval_sandbox.cli import main
        print("✅ CLI imported successfully")
        return True
        
    except ImportError as e:
        print(f"❌ CLI import failed: {e}")
        return False


def test_exceptions():
    """Test that custom exceptions work."""
    print("\n🧪 Testing exception handling...")
    
    try:
        from agi_eval_sandbox.core.exceptions import ValidationError, ConfigurationError
        
        # Test ValidationError
        try:
            raise ValidationError("Test validation error", {"field": "test"})
        except ValidationError as e:
            assert str(e) == "Test validation error"
            assert e.details == {"field": "test"}
            print("✅ ValidationError works correctly")
        
        # Test ConfigurationError
        try:
            raise ConfigurationError("Test config error")
        except ConfigurationError as e:
            assert str(e) == "Test config error"
            print("✅ ConfigurationError works correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Exception test failed: {e}")
        return False


def test_settings():
    """Test settings configuration."""
    print("\n🧪 Testing settings...")
    
    try:
        from agi_eval_sandbox.config.settings import Settings
        
        settings = Settings()
        
        # Check that settings object exists and has expected attributes
        assert hasattr(settings, 'openai_api_key')
        assert hasattr(settings, 'anthropic_api_key')
        assert hasattr(settings, 'debug')
        
        print("✅ Settings configuration works")
        return True
        
    except Exception as e:
        print(f"❌ Settings test failed: {e}")
        return False


def test_project_structure():
    """Test that project structure is correct."""
    print("\n🧪 Testing project structure...")
    
    required_files = [
        "src/agi_eval_sandbox/__init__.py",
        "src/agi_eval_sandbox/cli.py",
        "src/agi_eval_sandbox/core/__init__.py",
        "src/agi_eval_sandbox/core/context_compressor.py",
        "src/agi_eval_sandbox/core/advanced_compressor.py",
        "src/agi_eval_sandbox/api/main.py",
        "src/agi_eval_sandbox/config/settings.py",
        "pyproject.toml",
        "README.md"
    ]
    
    missing_files = []
    
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All required files present")
        return True


def test_compression_enums():
    """Test compression strategy enums without dependencies."""
    print("\n🧪 Testing compression enums...")
    
    try:
        # Import without initializing heavy dependencies
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "context_compressor", 
            "src/agi_eval_sandbox/core/context_compressor.py"
        )
        
        # Check if the file has the required enums
        with open("src/agi_eval_sandbox/core/context_compressor.py", 'r') as f:
            content = f.read()
            
        required_patterns = [
            "class CompressionStrategy(Enum)",
            "EXTRACTIVE_SUMMARIZATION",
            "SENTENCE_CLUSTERING", 
            "SEMANTIC_FILTERING",
            "TOKEN_PRUNING",
            "IMPORTANCE_SAMPLING",
            "HIERARCHICAL_COMPRESSION",
            "class CompressionMetrics",
            "class CompressionConfig",
            "class ContextCompressionEngine"
        ]
        
        for pattern in required_patterns:
            if pattern not in content:
                print(f"❌ Missing pattern: {pattern}")
                return False
        
        print("✅ Compression enums and classes defined correctly")
        return True
        
    except Exception as e:
        print(f"❌ Compression enum test failed: {e}")
        return False


def run_basic_tests():
    """Run all basic tests."""
    print("🚀 Starting Basic Functionality Tests")
    print("="*50)
    
    tests = [
        test_import,
        test_exceptions,
        test_settings, 
        test_project_structure,
        test_compression_enums,
        test_cli_import,
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            success = test_func()
            if success:
                passed_tests += 1
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed with exception: {e}")
    
    print(f"\n{'='*50}")
    print("🏁 BASIC TEST RESULTS")
    print(f"{'='*50}")
    print(f"✅ Passed: {passed_tests}/{total_tests} tests")
    print(f"📊 Success rate: {(passed_tests/total_tests*100):.1f}%")
    
    if passed_tests == total_tests:
        print("🎉 All basic tests passed!")
        return True
    else:
        print("⚠️  Some basic tests failed.")
        return False


def main():
    """Main test function."""
    print("AGI Evaluation Sandbox - Basic Functionality Test Suite")
    print("======================================================")
    
    success = run_basic_tests()
    
    if success:
        print("\n✅ Basic functionality tests completed successfully!")
        print("Next: Install dependencies to run full tests:")
        print("pip install numpy scikit-learn sentence-transformers torch")
        return 0
    else:
        print("\n❌ Some basic functionality tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)