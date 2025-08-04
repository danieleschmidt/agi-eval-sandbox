#!/usr/bin/env python3
"""
Test Generation 2 robust implementation features.
"""

import sys
import os
import asyncio
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_validation():
    """Test input validation and security features."""
    from agi_eval_sandbox.core.validation import InputValidator
    from agi_eval_sandbox.core.exceptions import ValidationError, SecurityError
    
    print("🧪 Testing Input Validation")
    print("-" * 30)
    
    # Test model name validation
    try:
        valid_name = InputValidator.validate_model_name("gpt-4")
        assert valid_name == "gpt-4"
        print("✅ Valid model name accepted")
    except Exception as e:
        print(f"❌ Model name validation failed: {e}")
        return False
    
    # Test invalid model name
    try:
        InputValidator.validate_model_name("<script>alert('xss')</script>")
        print("❌ Security validation failed - dangerous input not blocked")
        return False
    except SecurityError:
        print("✅ Security validation working - dangerous input blocked")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Test provider validation
    try:
        valid_provider = InputValidator.validate_provider("OpenAI")
        assert valid_provider == "openai"
        print("✅ Provider validation working")
    except Exception as e:
        print(f"❌ Provider validation failed: {e}")
        return False
    
    # Test temperature validation
    try:
        valid_temp = InputValidator.validate_temperature(0.5)
        assert valid_temp == 0.5
        print("✅ Temperature validation working")
    except Exception as e:
        print(f"❌ Temperature validation failed: {e}")
        return False
    
    # Test invalid temperature
    try:
        InputValidator.validate_temperature(3.0)
        print("❌ Temperature validation failed - invalid value accepted")
        return False
    except ValidationError:
        print("✅ Temperature validation working - invalid value rejected")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False
    
    # Test input sanitization
    try:
        dangerous_input = "<script>alert('xss')</script>Hello World"
        sanitized = InputValidator.sanitize_user_input(dangerous_input)
        assert "<script>" not in sanitized
        assert "Hello World" in sanitized
        print("✅ Input sanitization working")
    except Exception as e:
        print(f"❌ Input sanitization failed: {e}")
        return False
    
    return True

def test_exceptions():
    """Test custom exception handling."""
    from agi_eval_sandbox.core.exceptions import (
        AGIEvalError, ValidationError, ModelProviderError, SecurityError
    )
    
    print("\n🧪 Testing Exception Handling")
    print("-" * 30)
    
    # Test base exception
    try:
        error = AGIEvalError("Test error", {"detail": "test"})
        assert error.message == "Test error"
        assert error.details["detail"] == "test"
        print("✅ Base exception working")
    except Exception as e:
        print(f"❌ Base exception failed: {e}")
        return False
    
    # Test specific exceptions
    try:
        validation_error = ValidationError("Invalid input")
        provider_error = ModelProviderError("API failed")
        security_error = SecurityError("Security violation")
        
        assert isinstance(validation_error, AGIEvalError)
        assert isinstance(provider_error, AGIEvalError)
        assert isinstance(security_error, AGIEvalError)
        print("✅ Specific exceptions working")
    except Exception as e:
        print(f"❌ Specific exceptions failed: {e}")
        return False
    
    return True

def test_logging():
    """Test logging configuration."""  
    from agi_eval_sandbox.core.logging_config import (
        get_logger, setup_logging, security_logger, performance_logger
    )
    
    print("\n🧪 Testing Logging System")
    print("-" * 30)
    
    try:
        # Setup logging
        setup_logging(level="INFO", structured=True, include_console=False)
        print("✅ Logging setup successful")
        
        # Test logger creation
        logger = get_logger("test")
        assert logger.name == "agi_eval_sandbox.test"
        print("✅ Logger creation working")
        
        # Test specialized loggers
        security_logger.log_suspicious_activity("test_activity", {"detail": "test"})
        performance_logger.log_resource_usage(100.0, 50.0, 1000.0)
        print("✅ Specialized loggers working")
        
        return True
    except Exception as e:
        print(f"❌ Logging test failed: {e}")
        return False

async def test_health():
    """Test health monitoring."""
    from agi_eval_sandbox.core.health import health_monitor, HealthStatus
    
    print("\n🧪 Testing Health Monitoring")
    print("-" * 30)
    
    try:
        # Collect metrics
        metrics = health_monitor.collect_metrics()
        assert metrics.cpu_percent >= 0
        assert metrics.memory_percent >= 0
        print("✅ Metrics collection working")
        
        # Run health checks (async)
        checks = await health_monitor.run_all_checks()
        assert len(checks) > 0
        
        # Check overall health
        overall_status = health_monitor.get_overall_health()
        assert isinstance(overall_status, HealthStatus)
        
        # Get health summary
        summary = health_monitor.get_health_summary()
        assert "overall_status" in summary
        assert "checks" in summary
        assert "metrics" in summary
        
        print("✅ Health monitoring working")
        return True
            
    except Exception as e:
        print(f"❌ Health monitoring test failed: {e}")
        return False

async def test_robust_model():
    """Test robust model implementation."""
    from agi_eval_sandbox.core.models import Model, ModelConfig
    from agi_eval_sandbox.core.exceptions import ValidationError
    
    print("\n🧪 Testing Robust Model Implementation")
    print("-" * 30)
    
    try:
        # Test model config validation
        try:
            invalid_config = ModelConfig(
                name="",  # Invalid empty name
                provider="invalid",  # Invalid provider
                temperature=3.0  # Invalid temperature
            )
            print("❌ Model config validation failed - invalid config accepted")
            return False
        except ValidationError:
            print("✅ Model config validation working - invalid config rejected")
        
        # Test valid model creation
        model = Model(provider="local", name="test-model")
        assert model.name == "test-model"
        assert model.provider_name == "local"
        print("✅ Valid model creation working")
        
        # Test robust generation with validation
        response = await model.generate("What is 2+2?")
        assert isinstance(response, str)
        assert len(response) > 0
        print("✅ Robust model generation working")
        
        return True
        
    except Exception as e:
        print(f"❌ Robust model test failed: {e}")
        return False

async def main():
    """Run all Generation 2 tests."""
    print("🚀 Generation 2 (Robust) Implementation Tests")
    print("=" * 60)
    
    tests = [
        ("Validation", test_validation, False),
        ("Exceptions", test_exceptions, False),
        ("Logging", test_logging, False),     
        ("Health", test_health, True),  # Async test
        ("Robust Models", test_robust_model, True),  # Async test
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func, is_async in tests:
        print(f"\n📋 Testing {name}:")
        try:
            if is_async:
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                print(f"✅ {name} tests passed")
            else:
                print(f"❌ {name} tests failed")
        except Exception as e:
            print(f"❌ {name} tests failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"🎯 Generation 2 Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 Generation 2 (Robust) implementation is working!")
        print("✅ Comprehensive error handling implemented")
        print("✅ Input validation and security measures active")
        print("✅ Structured logging and monitoring in place")
        print("✅ Health checks and system monitoring functional")
        return True
    else:
        print("💥 Some Generation 2 tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)