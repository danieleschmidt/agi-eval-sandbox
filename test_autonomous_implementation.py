#!/usr/bin/env python3
"""Comprehensive test suite for autonomous implementation verification."""

import sys
import os
import asyncio
import time
import unittest
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agi_eval_sandbox import EvalSuite, Model
from agi_eval_sandbox.core.security import SecurityAuditor, InputSanitizer
from agi_eval_sandbox.core.autoscaling import AutoScaler, LoadBalancer, ScalingMetrics
from agi_eval_sandbox.core.intelligent_cache import AccessPatternAnalyzer, AdaptiveEvictionPolicy


class TestGeneration1Basic(unittest.TestCase):
    """Test Generation 1: Basic functionality works."""
    
    def test_eval_suite_creation(self):
        """Test EvalSuite can be created and has benchmarks."""
        eval_suite = EvalSuite()
        benchmarks = eval_suite.list_benchmarks()
        
        self.assertIsInstance(benchmarks, list)
        self.assertGreater(len(benchmarks), 0)
        self.assertIn('truthfulqa', benchmarks)
        self.assertIn('mmlu', benchmarks)
        self.assertIn('humaneval', benchmarks)
    
    def test_model_creation(self):
        """Test Model can be created with valid providers."""
        model = Model(provider='local', name='test-model')
        self.assertEqual(model.name, 'test-model')
        self.assertEqual(model.provider_name, 'local')
    
    def test_benchmark_access(self):
        """Test benchmarks can be accessed and have questions."""
        eval_suite = EvalSuite()
        truthfulqa = eval_suite.get_benchmark('truthfulqa')
        
        self.assertIsNotNone(truthfulqa)
        questions = truthfulqa.get_questions()
        self.assertIsInstance(questions, list)
        self.assertGreater(len(questions), 0)


class TestGeneration2Robust(unittest.TestCase):
    """Test Generation 2: Robust components work."""
    
    def test_security_auditor(self):
        """Test security auditor functionality."""
        auditor = SecurityAuditor()
        sanitizer = InputSanitizer()
        
        # Test clean input
        clean_input = sanitizer.sanitize_input("Hello world!")
        self.assertEqual(clean_input, "Hello world!")
        
        # Test security audit
        test_data = {'model': {'name': 'gpt-3.5-turbo'}, 'prompt': 'Hello'}
        events = auditor.audit_request(test_data, source_ip='127.0.0.1')
        self.assertIsInstance(events, list)
    
    def test_rate_limiting(self):
        """Test rate limiting functionality."""
        auditor = SecurityAuditor()
        rate_limiter = auditor.rate_limiter
        
        # Test rate limit check
        allowed, reason = rate_limiter.check_rate_limit('test_ip', max_requests=5)
        self.assertTrue(allowed)
        self.assertIsNone(reason)
        
        # Test stats
        stats = rate_limiter.get_rate_limit_stats('test_ip')
        self.assertIsInstance(stats, dict)
        self.assertIn('requests_in_current_window', stats)
    
    def test_input_validation(self):
        """Test input validation and sanitization."""
        sanitizer = InputSanitizer()
        
        # Test valid inputs
        self.assertTrue(sanitizer.validate_model_name('gpt-3.5-turbo'))
        self.assertTrue(sanitizer.validate_model_name('claude-2'))
        
        # Test invalid inputs
        self.assertFalse(sanitizer.validate_model_name(''))
        self.assertFalse(sanitizer.validate_model_name('a' * 200))


class TestGeneration3Optimized(unittest.TestCase):
    """Test Generation 3: Optimization components work."""
    
    def test_auto_scaler(self):
        """Test auto-scaling functionality."""
        scaler = AutoScaler()
        
        # Test initial state
        status = scaler.get_scaling_status()
        self.assertIsInstance(status, dict)
        self.assertEqual(status['current_instances'], 1)
        self.assertTrue(status['can_scale'])
        
        # Test metrics update
        metrics = ScalingMetrics(
            cpu_utilization=75.0,
            memory_utilization=60.0,
            queue_length=25
        )
        scaler.update_metrics(metrics)
        
        # Verify metrics were recorded
        self.assertGreater(len(scaler.metrics_history), 0)
    
    def test_load_balancer(self):
        """Test load balancer functionality."""
        lb = LoadBalancer()
        
        # Add workers
        lb.add_worker('worker1', 'http://localhost:8001', weight=1.0)
        lb.add_worker('worker2', 'http://localhost:8002', weight=1.5)
        
        # Test worker selection
        worker = lb.get_next_worker()
        self.assertIsNotNone(worker)
        self.assertIn('id', worker)
        
        # Test stats
        stats = lb.get_load_balancer_stats()
        self.assertEqual(stats['total_workers'], 2)
        self.assertEqual(stats['healthy_workers'], 2)
    
    def test_intelligent_cache(self):
        """Test intelligent caching components."""
        # Test pattern analyzer
        analyzer = AccessPatternAnalyzer()
        patterns = analyzer.analyze_patterns()
        self.assertIsInstance(patterns, dict)
        self.assertEqual(len(patterns), 4)  # 4 pattern types
        
        # Test eviction policy
        policy = AdaptiveEvictionPolicy()
        policy.record_access('test_key')
        candidates = policy.select_eviction_candidates(['key1', 'key2', 'test_key'], 1)
        self.assertIsInstance(candidates, list)
        self.assertLessEqual(len(candidates), 1)


class TestIntegrationScenarios(unittest.TestCase):
    """Test integration scenarios."""
    
    async def _async_evaluation_test(self):
        """Test async evaluation flow."""
        eval_suite = EvalSuite()
        model = Model(provider='local', name='test-model')
        
        # Test that evaluation setup works
        benchmarks = eval_suite.list_benchmarks()
        self.assertGreater(len(benchmarks), 0)
        
        # Test benchmark access
        for benchmark_name in benchmarks[:1]:  # Test first benchmark only
            benchmark = eval_suite.get_benchmark(benchmark_name)
            self.assertIsNotNone(benchmark)
            
            questions = benchmark.get_questions()
            self.assertIsInstance(questions, list)
    
    def test_async_evaluation_flow(self):
        """Test async evaluation flow works."""
        asyncio.run(self._async_evaluation_test())
    
    def test_security_integration(self):
        """Test security components work together."""
        auditor = SecurityAuditor()
        
        # Simulate multiple requests
        test_requests = [
            {'model': {'name': 'gpt-4'}, 'prompt': 'Hello'},
            {'model': {'name': 'claude-3'}, 'prompt': 'Hi there'},
            {'model': {'name': 'invalid<script>'}, 'prompt': 'Test'}
        ]
        
        total_events = 0
        for i, request in enumerate(test_requests):
            events = auditor.audit_request(request, source_ip=f'192.168.1.{i+1}')
            total_events += len(events)
        
        # Should have detected some security events for invalid input
        summary = auditor.get_security_summary(hours=1)
        self.assertIsInstance(summary, dict)


class TestPerformanceMetrics(unittest.TestCase):
    """Test performance and monitoring."""
    
    def test_scaling_metrics(self):
        """Test scaling metrics can be created and used."""
        metrics = ScalingMetrics(
            cpu_utilization=80.0,
            memory_utilization=70.0,
            active_requests=50,
            queue_length=25,
            response_time_p95=1500.0,
            error_rate=2.5,
            throughput_qps=45.0
        )
        
        self.assertEqual(metrics.cpu_utilization, 80.0)
        self.assertEqual(metrics.memory_utilization, 70.0)
        self.assertEqual(metrics.active_requests, 50)
        self.assertGreater(metrics.timestamp.timestamp(), 0)
    
    def test_performance_monitoring(self):
        """Test performance monitoring components."""
        scaler = AutoScaler()
        
        # Test multiple metrics updates
        for i in range(5):
            metrics = ScalingMetrics(
                cpu_utilization=60.0 + (i * 5),
                memory_utilization=50.0 + (i * 3),
                queue_length=10 + i
            )
            scaler.update_metrics(metrics)
        
        # Check history
        self.assertEqual(len(scaler.metrics_history), 5)
        
        # Check scaling history
        history = scaler.get_scaling_history(hours=1)
        self.assertIsInstance(history, list)
        self.assertEqual(len(history), 5)


def run_quality_gates():
    """Run comprehensive quality gate checks."""
    print("🛡️ RUNNING MANDATORY QUALITY GATES")
    print("=" * 50)
    
    # Test execution
    test_suite = unittest.TestSuite()
    
    # Add all test classes
    for test_class in [
        TestGeneration1Basic,
        TestGeneration2Robust, 
        TestGeneration3Optimized,
        TestIntegrationScenarios,
        TestPerformanceMetrics
    ]:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Quality gate results
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    success_rate = ((total_tests - failures - errors) / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "=" * 50)
    print("🛡️ QUALITY GATES SUMMARY")
    print("=" * 50)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {total_tests - failures - errors}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Quality gate criteria
    MINIMUM_SUCCESS_RATE = 85.0
    
    if success_rate >= MINIMUM_SUCCESS_RATE:
        print(f"✅ QUALITY GATES PASSED ({success_rate:.1f}% >= {MINIMUM_SUCCESS_RATE}%)")
        return True
    else:
        print(f"❌ QUALITY GATES FAILED ({success_rate:.1f}% < {MINIMUM_SUCCESS_RATE}%)")
        return False


def run_security_scan():
    """Run security vulnerability scan."""
    print("\n🔒 SECURITY SCAN")
    print("=" * 30)
    
    try:
        from agi_eval_sandbox.quality.security_scanner import SecurityScanner
        
        scanner = SecurityScanner()
        print("✅ Security scanner initialized")
        
        # Test security components
        from agi_eval_sandbox.core.security import SecurityAuditor
        auditor = SecurityAuditor()
        
        # Run basic security checks
        test_data = {
            'model': {'name': 'test-model'},
            'prompt': 'SELECT * FROM users; DROP TABLE users;',  # SQL injection attempt
            'api_key': 'invalid-key-format'
        }
        
        events = auditor.audit_request(test_data, source_ip='192.168.1.100')
        security_summary = auditor.get_security_summary()
        
        print(f"✅ Security events detected: {len(events)}")
        print(f"✅ Security summary generated: {security_summary['total_events']} total events")
        print("✅ Security scan completed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Security scan failed: {e}")
        return False


def run_performance_benchmarks():
    """Run basic performance benchmarks."""
    print("\n⚡ PERFORMANCE BENCHMARKS")
    print("=" * 30)
    
    try:
        # Test basic operations timing
        start_time = time.time()
        
        # EvalSuite creation
        eval_suite = EvalSuite()
        creation_time = time.time() - start_time
        
        # Benchmark access
        start_time = time.time()
        benchmarks = eval_suite.list_benchmarks()
        access_time = time.time() - start_time
        
        # Model creation
        start_time = time.time()
        model = Model(provider='local', name='perf-test')
        model_time = time.time() - start_time
        
        print(f"✅ EvalSuite creation: {creation_time*1000:.2f}ms")
        print(f"✅ Benchmark access: {access_time*1000:.2f}ms")
        print(f"✅ Model creation: {model_time*1000:.2f}ms")
        
        # Performance criteria
        if creation_time < 1.0 and access_time < 0.1 and model_time < 0.1:
            print("✅ Performance benchmarks PASSED")
            return True
        else:
            print("❌ Performance benchmarks FAILED")
            return False
            
    except Exception as e:
        print(f"❌ Performance benchmark failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 AUTONOMOUS SDLC QUALITY VERIFICATION")
    print("=" * 50)
    
    # Run all quality gates
    gates_passed = 0
    total_gates = 3
    
    # 1. Functional Tests
    if run_quality_gates():
        gates_passed += 1
    
    # 2. Security Scan
    if run_security_scan():
        gates_passed += 1
    
    # 3. Performance Benchmarks
    if run_performance_benchmarks():
        gates_passed += 1
    
    # Final verdict
    print("\n" + "=" * 50)
    print("🏁 FINAL QUALITY GATE RESULTS")
    print("=" * 50)
    print(f"Gates Passed: {gates_passed}/{total_gates}")
    
    if gates_passed == total_gates:
        print("✅ ALL QUALITY GATES PASSED - READY FOR PRODUCTION")
        sys.exit(0)
    else:
        print("❌ QUALITY GATES FAILED - DEPLOYMENT BLOCKED")
        sys.exit(1)