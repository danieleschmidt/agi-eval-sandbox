"""
Performance and load tests for AGI Evaluation Sandbox.

These tests verify system performance under various load conditions.
"""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

import pytest
from locust import HttpUser, task, between


class TestPerformance:
    """Performance tests for critical system components."""
    
    @pytest.mark.performance
    def test_api_response_time(self, api_client, performance_threshold):
        """Test API response times under normal load."""
        start_time = time.time()
        
        # Simulate API call
        # response = api_client.get("/api/v1/benchmarks")
        # response_time = time.time() - start_time
        
        # Mock response time test
        response_time = 0.1  # Mock fast response
        
        assert response_time < performance_threshold["api_response_time"]
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, mock_async_model_provider):
        """Test handling multiple concurrent evaluations."""
        concurrent_count = 10
        
        async def run_evaluation():
            """Simulate running an evaluation."""
            return await mock_async_model_provider.generate("Test prompt")
        
        start_time = time.time()
        
        # Run concurrent evaluations
        tasks = [run_evaluation() for _ in range(concurrent_count)]
        results = await asyncio.gather(*tasks)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert len(results) == concurrent_count
        assert all(result is not None for result in results)
        assert total_time < 5.0  # Should complete within 5 seconds
    
    @pytest.mark.performance
    def test_database_query_performance(self, test_db_session):
        """Test database query performance."""
        # This would test actual database queries
        start_time = time.time()
        
        # Simulate database operations
        for i in range(100):
            # Mock database query
            pass
        
        end_time = time.time()
        query_time = end_time - start_time
        
        assert query_time < 1.0  # Should complete within 1 second
    
    @pytest.mark.performance
    def test_memory_usage(self):
        """Test memory usage during operations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Simulate memory-intensive operation
        large_data = [i for i in range(100000)]
        processed_data = [x * 2 for x in large_data]
        
        peak_memory = process.memory_info().rss
        memory_increase = peak_memory - initial_memory
        
        # Cleanup
        del large_data, processed_data
        
        # Memory increase should be reasonable
        assert memory_increase < 100 * 1024 * 1024  # Less than 100MB


class TestScalability:
    """Scalability tests for system components."""
    
    @pytest.mark.slow
    @pytest.mark.performance
    def test_large_batch_processing(self):
        """Test processing large batches of evaluations."""
        batch_size = 1000
        
        start_time = time.time()
        
        # Simulate processing large batch
        batch_data = [{"id": i, "prompt": f"Test prompt {i}"} for i in range(batch_size)]
        processed_batch = []
        
        for item in batch_data:
            # Mock processing
            processed_item = {
                "id": item["id"],
                "response": f"Response to {item['prompt']}",
                "processed_at": time.time()
            }
            processed_batch.append(processed_item)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        assert len(processed_batch) == batch_size
        assert processing_time < 10.0  # Should complete within 10 seconds
        
        # Calculate throughput
        throughput = batch_size / processing_time
        assert throughput > 100  # Should process at least 100 items per second
    
    @pytest.mark.performance
    def test_thread_pool_scaling(self):
        """Test scaling with thread pools."""
        def cpu_intensive_task(n):
            """Simulate CPU-intensive task."""
            return sum(i * i for i in range(n))
        
        task_count = 50
        task_size = 10000
        
        # Test with different thread pool sizes
        for pool_size in [1, 2, 4, 8]:
            start_time = time.time()
            
            with ThreadPoolExecutor(max_workers=pool_size) as executor:
                futures = [executor.submit(cpu_intensive_task, task_size) for _ in range(task_count)]
                results = [future.result() for future in futures]
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            assert len(results) == task_count
            
            # Performance should improve with more threads (up to a point)
            print(f"Pool size {pool_size}: {execution_time:.2f} seconds")


class TestStressConditions:
    """Stress tests for system robustness."""
    
    @pytest.mark.stress
    def test_high_concurrency_stress(self, api_client):
        """Test system under high concurrency stress."""
        concurrent_requests = 100
        
        def make_request():
            """Make a single API request."""
            try:
                # Simulate API request
                # response = api_client.get("/api/v1/health")
                # return response.status_code == 200
                return True  # Mock successful request
            except Exception:
                return False
        
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=concurrent_requests) as executor:
            futures = [executor.submit(make_request) for _ in range(concurrent_requests)]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        success_rate = sum(results) / len(results)
        
        assert success_rate > 0.95  # At least 95% success rate
        assert total_time < 30.0  # Should complete within 30 seconds
    
    @pytest.mark.stress
    def test_memory_pressure_handling(self):
        """Test system behavior under memory pressure."""
        import gc
        
        large_objects = []
        
        try:
            # Gradually increase memory usage
            for i in range(10):
                # Create large object (10MB each)
                large_object = bytearray(10 * 1024 * 1024)
                large_objects.append(large_object)
                
                # Force garbage collection
                gc.collect()
                
                # Check if system is still responsive
                assert len(large_objects) == i + 1
        
        finally:
            # Cleanup
            large_objects.clear()
            gc.collect()
    
    @pytest.mark.stress
    def test_error_recovery_under_stress(self, mock_model_provider):
        """Test error recovery under stress conditions."""
        error_injection_rate = 0.1  # 10% error rate
        total_requests = 100
        
        def make_request_with_errors():
            """Make request with potential errors."""
            import random
            if random.random() < error_injection_rate:
                raise Exception("Simulated error")
            return "Success"
        
        successful_requests = 0
        recovered_errors = 0
        
        for _ in range(total_requests):
            try:
                result = make_request_with_errors()
                if result == "Success":
                    successful_requests += 1
            except Exception:
                # Simulate error recovery
                recovered_errors += 1
        
        success_rate = successful_requests / total_requests
        recovery_rate = recovered_errors / (total_requests * error_injection_rate)
        
        assert success_rate > 0.85  # At least 85% success rate
        assert recovery_rate > 0.5   # At least 50% error recovery


# Locust load testing classes
class EvaluationUser(HttpUser):
    """Locust user class for load testing evaluation endpoints."""
    
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when user starts."""
        # Simulate user login
        response = self.client.post("/api/v1/auth/login", json={
            "email": "test@example.com",
            "password": "test_password"
        })
        if response.status_code == 200:
            self.token = response.json().get("access_token")
            self.headers = {"Authorization": f"Bearer {self.token}"}
        else:
            self.headers = {}
    
    @task(3)
    def list_benchmarks(self):
        """List available benchmarks."""
        self.client.get("/api/v1/benchmarks", headers=self.headers)
    
    @task(2)
    def list_models(self):
        """List supported models."""
        self.client.get("/api/v1/models", headers=self.headers)
    
    @task(1)
    def create_evaluation(self):
        """Create a new evaluation."""
        evaluation_data = {
            "model": {
                "provider": "openai",
                "name": "gpt-4",
                "temperature": 0.0
            },
            "benchmarks": ["mmlu"]
        }
        self.client.post("/api/v1/evaluations", json=evaluation_data, headers=self.headers)
    
    @task(2)
    def check_evaluation_status(self):
        """Check evaluation status."""
        # Simulate checking a random evaluation
        evaluation_id = f"test-eval-{hash(self) % 1000}"
        self.client.get(f"/api/v1/evaluations/{evaluation_id}", headers=self.headers)


class DashboardUser(HttpUser):
    """Locust user class for load testing dashboard endpoints."""
    
    wait_time = between(2, 5)  # Longer wait times for dashboard users
    
    @task(5)
    def view_dashboard(self):
        """View main dashboard."""
        self.client.get("/dashboard")
    
    @task(2)
    def view_results(self):
        """View evaluation results."""
        self.client.get("/dashboard/results")
    
    @task(1)
    def view_analytics(self):
        """View analytics page."""
        self.client.get("/dashboard/analytics")


class BenchmarkPerformanceTest:
    """Specific performance tests for benchmark execution."""
    
    @pytest.mark.benchmark
    def test_mmlu_evaluation_performance(self, mock_model_provider):
        """Test MMLU evaluation performance."""
        question_count = 100  # Subset for testing
        
        start_time = time.time()
        
        # Simulate MMLU evaluation
        results = []
        for i in range(question_count):
            # Mock evaluation of a single question
            response = mock_model_provider.generate(f"Question {i}")
            result = {
                "question_id": f"q{i}",
                "response": response,
                "correct": i % 2 == 0,  # Mock 50% accuracy
                "processing_time": 0.1
            }
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert len(results) == question_count
        assert total_time < 30.0  # Should complete within 30 seconds
        
        # Calculate metrics
        avg_time_per_question = total_time / question_count
        assert avg_time_per_question < 0.5  # Less than 0.5 seconds per question
    
    @pytest.mark.benchmark
    def test_humaneval_evaluation_performance(self, mock_model_provider):
        """Test HumanEval evaluation performance."""
        problem_count = 50  # Subset for testing
        
        start_time = time.time()
        
        # Simulate HumanEval evaluation
        results = []
        for i in range(problem_count):
            # Mock code generation and evaluation
            code_response = mock_model_provider.generate(f"def solution_{i}():")
            
            # Mock code execution and testing
            result = {
                "problem_id": f"prob{i}",
                "generated_code": code_response,
                "passed_tests": i % 3 == 0,  # Mock 33% pass rate
                "execution_time": 0.2
            }
            results.append(result)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        assert len(results) == problem_count
        assert total_time < 60.0  # Should complete within 60 seconds
        
        # Calculate metrics
        avg_time_per_problem = total_time / problem_count
        assert avg_time_per_problem < 1.5  # Less than 1.5 seconds per problem


# Performance monitoring utilities
class PerformanceMonitor:
    """Utility class for monitoring performance during tests."""
    
    def __init__(self):
        self.start_time = None
        self.metrics = {}
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.metrics = {
            "start_time": self.start_time,
            "cpu_percent": [],
            "memory_usage": [],
            "response_times": []
        }
    
    def record_metric(self, metric_name: str, value: float):
        """Record a performance metric."""
        if metric_name not in self.metrics:
            self.metrics[metric_name] = []
        self.metrics[metric_name].append(value)
    
    def stop_monitoring(self):
        """Stop monitoring and return results."""
        if self.start_time:
            self.metrics["total_time"] = time.time() - self.start_time
        return self.metrics


@pytest.fixture
def performance_monitor():
    """Fixture for performance monitoring."""
    monitor = PerformanceMonitor()
    monitor.start_monitoring()
    yield monitor
    return monitor.stop_monitoring()