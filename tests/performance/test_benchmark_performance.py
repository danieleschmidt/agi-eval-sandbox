"""
Performance tests for benchmark evaluation and system throughput.
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from agi_eval_sandbox.core import EvalSuite, Model
from agi_eval_sandbox.core.benchmarks import CustomBenchmark, Question, QuestionType
from agi_eval_sandbox.core.models import create_mock_model


@pytest.mark.performance
class TestBenchmarkEvaluationPerformance:
    """Test performance characteristics of benchmark evaluation."""
    
    @pytest.mark.asyncio
    async def test_single_question_evaluation_speed(self, performance_threshold):
        """Test speed of evaluating a single question."""
        # Use local mock model for consistent timing
        model = create_mock_model("speed-test", simulate_delay=0.01)
        suite = EvalSuite()
        
        # Create a simple custom benchmark with one question
        questions = [
            Question(
                id="speed-test-1",
                prompt="What is 2+2?",
                correct_answer="4",
                question_type=QuestionType.SHORT_ANSWER
            )
        ]
        
        benchmark = CustomBenchmark("speed-test", questions)
        suite.register_benchmark(benchmark)
        
        # Time the evaluation
        start_time = time.time()
        
        results = await suite.evaluate(
            model=model,
            benchmarks=["speed-test"],
            num_questions=1,
            save_results=False,
            parallel=False
        )
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete quickly (within performance threshold)
        assert duration < performance_threshold["evaluation_time"] / 100  # Much faster for single question
        assert results.summary()["total_questions"] == 1
    
    @pytest.mark.asyncio
    async def test_batch_evaluation_scalability(self, performance_threshold):
        """Test performance scaling with increasing number of questions."""
        model = create_mock_model("batch-test", simulate_delay=0.005)
        suite = EvalSuite()
        
        question_counts = [10, 50, 100]
        durations = []
        
        for count in question_counts:
            # Create benchmark with specified number of questions
            questions = [
                Question(
                    id=f"batch-q{i}",
                    prompt=f"Question {i}: What is {i}+1?",
                    correct_answer=str(i + 1),
                    question_type=QuestionType.SHORT_ANSWER
                )
                for i in range(count)
            ]
            
            benchmark = CustomBenchmark(f"batch-test-{count}", questions)
            suite.register_benchmark(benchmark)
            
            start_time = time.time()
            
            results = await suite.evaluate(
                model=model,
                benchmarks=[f"batch-test-{count}"],
                save_results=False,
                parallel=True
            )
            
            end_time = time.time()
            duration = end_time - start_time
            durations.append(duration)
            
            # Verify results
            assert results.summary()["total_questions"] == count
            
            # Performance check - should not grow linearly with question count
            # due to parallelization
            if count == 100:
                assert duration < performance_threshold["evaluation_time"]
        
        # Check that scaling is sub-linear (parallelization working)
        # 10x questions should not take 10x time
        assert durations[2] < durations[0] * 5  # 100 questions < 10 questions * 5
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluations(self, performance_threshold):
        """Test performance of concurrent evaluations."""
        # Create multiple models for concurrent testing
        models = [
            create_mock_model(f"concurrent-model-{i}", simulate_delay=0.01)
            for i in range(5)
        ]
        
        suite = EvalSuite()
        
        # Create a small benchmark for testing
        questions = [
            Question(
                id=f"concurrent-q{i}",
                prompt=f"Concurrent question {i}",
                correct_answer=f"Answer {i}",
                question_type=QuestionType.SHORT_ANSWER
            )
            for i in range(10)
        ]
        
        benchmark = CustomBenchmark("concurrent-test", questions)
        suite.register_benchmark(benchmark)
        
        async def run_evaluation(model):
            """Run evaluation for a single model."""
            return await suite.evaluate(
                model=model,
                benchmarks=["concurrent-test"],
                save_results=False,
                parallel=True
            )
        
        # Time concurrent evaluations
        start_time = time.time()
        
        # Run all evaluations concurrently
        tasks = [run_evaluation(model) for model in models]
        results_list = await asyncio.gather(*tasks)
        
        end_time = time.time()
        concurrent_duration = end_time - start_time
        
        # Time sequential evaluations for comparison
        start_time = time.time()
        
        sequential_results = []
        for model in models:
            result = await run_evaluation(model)
            sequential_results.append(result)
        
        end_time = time.time()
        sequential_duration = end_time - start_time
        
        # Concurrent should be faster than sequential
        assert concurrent_duration < sequential_duration
        
        # Both should produce same number of results
        assert len(results_list) == len(sequential_results) == 5
        
        # All evaluations should complete successfully
        for results in results_list:
            assert results.summary()["total_questions"] == 10
    
    @pytest.mark.asyncio
    async def test_memory_usage_during_evaluation(self):
        """Test memory usage during large evaluations."""
        import psutil
        import os
        
        # Get current process
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss
        
        # Create a large benchmark
        model = create_mock_model("memory-test", simulate_delay=0.001)
        suite = EvalSuite()
        
        questions = [
            Question(
                id=f"memory-q{i}",
                prompt=f"Memory test question {i} " + "padding " * 20,  # Add some content
                correct_answer=f"Answer {i}",
                question_type=QuestionType.SHORT_ANSWER
            )
            for i in range(500)
        ]
        
        benchmark = CustomBenchmark("memory-test", questions)
        suite.register_benchmark(benchmark)
        
        # Run evaluation
        results = await suite.evaluate(
            model=model,
            benchmarks=["memory-test"],
            save_results=False,
            parallel=True
        )
        
        # Measure peak memory during evaluation
        peak_memory = process.memory_info().rss
        
        # Clean up results
        del results
        del benchmark
        del questions
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Memory usage should be reasonable
        memory_increase = peak_memory - initial_memory
        memory_increase_mb = memory_increase / (1024 * 1024)
        
        # Should not use more than 512MB for 500 questions
        assert memory_increase_mb < 512, f"Memory usage too high: {memory_increase_mb:.1f}MB"
    
    def test_benchmark_loading_performance(self, performance_threshold):
        """Test performance of loading different benchmarks."""
        suite = EvalSuite()
        
        benchmark_names = suite.list_benchmarks()
        load_times = {}
        
        for benchmark_name in benchmark_names:
            start_time = time.time()
            
            benchmark = suite.get_benchmark(benchmark_name)
            questions = benchmark.get_questions()
            
            end_time = time.time()
            load_time = end_time - start_time
            load_times[benchmark_name] = load_time
            
            # Each benchmark should load reasonably quickly
            assert load_time < 5.0, f"Benchmark {benchmark_name} took {load_time:.2f}s to load"
            
            # Should have some questions
            assert len(questions) > 0
        
        # Average load time should be reasonable
        avg_load_time = statistics.mean(load_times.values())
        assert avg_load_time < 2.0, f"Average benchmark load time too high: {avg_load_time:.2f}s"
    
    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_performance(self):
        """Compare parallel vs sequential evaluation performance."""
        model = create_mock_model("parallel-test", simulate_delay=0.01)
        suite = EvalSuite()
        
        # Create benchmark with moderate number of questions
        questions = [
            Question(
                id=f"perf-q{i}",
                prompt=f"Performance question {i}",
                correct_answer=f"Answer {i}",
                question_type=QuestionType.SHORT_ANSWER
            )
            for i in range(50)
        ]
        
        benchmark = CustomBenchmark("perf-test", questions)
        suite.register_benchmark(benchmark)
        
        # Time parallel evaluation
        start_time = time.time()
        parallel_results = await suite.evaluate(
            model=model,
            benchmarks=["perf-test"],
            save_results=False,
            parallel=True
        )
        parallel_time = time.time() - start_time
        
        # Reset model call count for fair comparison
        model.provider.reset_call_count()
        
        # Time sequential evaluation
        start_time = time.time()
        sequential_results = await suite.evaluate(
            model=model,
            benchmarks=["perf-test"],
            save_results=False,
            parallel=False
        )
        sequential_time = time.time() - start_time
        
        # Parallel should be significantly faster
        speedup = sequential_time / parallel_time
        assert speedup > 2.0, f"Parallel speedup too low: {speedup:.2f}x"
        
        # Results should be equivalent
        assert parallel_results.summary()["total_questions"] == sequential_results.summary()["total_questions"]
    
    @pytest.mark.asyncio
    async def test_error_handling_performance_impact(self):
        """Test that error handling doesn't significantly impact performance."""
        # Create a model that fails sometimes
        failing_model = create_mock_model(
            "failing-test", 
            simulate_delay=0.005,
            simulate_failures=True,
            failure_rate=0.3  # 30% failure rate
        )
        
        successful_model = create_mock_model(
            "successful-test",
            simulate_delay=0.005,
            simulate_failures=False
        )
        
        suite = EvalSuite()
        
        questions = [
            Question(
                id=f"error-q{i}",
                prompt=f"Error test question {i}",
                correct_answer=f"Answer {i}",
                question_type=QuestionType.SHORT_ANSWER
            )
            for i in range(30)
        ]
        
        benchmark = CustomBenchmark("error-test", questions)
        suite.register_benchmark(benchmark)
        
        # Time evaluation with failing model
        start_time = time.time()
        failing_results = await suite.evaluate(
            model=failing_model,
            benchmarks=["error-test"],
            save_results=False,
            parallel=True
        )
        failing_time = time.time() - start_time
        
        # Time evaluation with successful model
        start_time = time.time()
        successful_results = await suite.evaluate(
            model=successful_model,
            benchmarks=["error-test"],
            save_results=False,
            parallel=True
        )
        successful_time = time.time() - start_time
        
        # Failing model should not be dramatically slower due to error handling
        performance_ratio = failing_time / successful_time
        assert performance_ratio < 3.0, f"Error handling overhead too high: {performance_ratio:.2f}x"
        
        # Both should process same number of questions
        assert failing_results.summary()["total_questions"] == successful_results.summary()["total_questions"]


@pytest.mark.performance
class TestAPIPerformance:
    """Test API endpoint performance characteristics."""
    
    def test_health_endpoint_performance(self, client):
        """Test health endpoint response time."""
        # Warm up
        client.get("/health")
        
        # Time multiple requests
        times = []
        for _ in range(10):
            start_time = time.time()
            response = client.get("/health")
            end_time = time.time()
            
            assert response.status_code == 200
            times.append(end_time - start_time)
        
        avg_response_time = statistics.mean(times)
        max_response_time = max(times)
        
        # Health endpoint should be very fast
        assert avg_response_time < 0.1, f"Health endpoint too slow: {avg_response_time:.3f}s average"
        assert max_response_time < 0.2, f"Health endpoint max time too slow: {max_response_time:.3f}s"
    
    def test_benchmark_listing_performance(self, client):
        """Test benchmark listing endpoint performance."""
        # Warm up
        client.get("/api/v1/benchmarks")
        
        # Time multiple requests
        times = []
        for _ in range(5):
            start_time = time.time()
            response = client.get("/api/v1/benchmarks")
            end_time = time.time()
            
            assert response.status_code == 200
            times.append(end_time - start_time)
        
        avg_response_time = statistics.mean(times)
        
        # Benchmark listing should be reasonably fast
        assert avg_response_time < 1.0, f"Benchmark listing too slow: {avg_response_time:.3f}s average"
    
    def test_concurrent_api_requests(self, client):
        """Test API performance under concurrent load."""
        def make_health_request():
            response = client.get("/health")
            return response.status_code == 200, time.time()
        
        def make_benchmark_request():
            response = client.get("/api/v1/benchmarks")
            return response.status_code == 200, time.time()
        
        # Mix of different endpoint requests
        request_functions = [make_health_request] * 10 + [make_benchmark_request] * 5
        
        start_time = time.time()
        
        # Execute concurrent requests
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(func) for func in request_functions]
            results = [future.result() for future in futures]
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Check that all requests succeeded
        success_count = sum(1 for success, _ in results if success)
        assert success_count == len(request_functions), f"Only {success_count}/{len(request_functions)} requests succeeded"
        
        # Should handle concurrent requests efficiently
        avg_time_per_request = total_time / len(request_functions)
        assert avg_time_per_request < 0.5, f"Concurrent request handling too slow: {avg_time_per_request:.3f}s per request"
    
    def test_large_custom_benchmark_creation_performance(self, client):
        """Test performance of creating large custom benchmarks."""
        # Create progressively larger benchmarks
        for size in [10, 50, 100]:
            questions = [
                {
                    "prompt": f"Performance test question {i}: What is the result of {i} + {i}?",
                    "correct_answer": str(i * 2),
                    "question_type": "short_answer",
                    "category": "math",
                    "difficulty": "easy"
                }
                for i in range(size)
            ]
            
            benchmark_data = {
                "name": f"Performance Test Benchmark {size}",
                "description": f"Benchmark with {size} questions for performance testing",
                "category": "performance_test",
                "questions": questions
            }
            
            start_time = time.time()
            response = client.post("/api/v1/benchmarks/custom", json=benchmark_data)
            end_time = time.time()
            
            creation_time = end_time - start_time
            
            assert response.status_code == 200
            data = response.json()
            assert data["total_questions"] == size
            
            # Should handle larger benchmarks efficiently
            # Time should not grow dramatically with size
            assert creation_time < size * 0.01, f"Benchmark creation too slow: {creation_time:.3f}s for {size} questions"
            
            # Clean up
            benchmark_id = data["benchmark_id"]
            client.delete(f"/api/v1/benchmarks/custom/{benchmark_id}")
    
    def test_stats_endpoint_performance(self, client):
        """Test statistics endpoint performance."""
        # Create some jobs first to have data
        for i in range(3):
            evaluation_request = {
                "model": {"provider": "local", "name": f"stats-perf-{i}"},
                "benchmarks": ["truthfulqa"],
                "config": {"temperature": 0.0, "num_questions": 1}
            }
            client.post("/api/v1/evaluate", json=evaluation_request)
        
        # Time stats endpoint
        times = []
        for _ in range(5):
            start_time = time.time()
            response = client.get("/api/v1/stats")
            end_time = time.time()
            
            assert response.status_code == 200
            times.append(end_time - start_time)
        
        avg_response_time = statistics.mean(times)
        
        # Stats should be computed quickly
        assert avg_response_time < 0.5, f"Stats endpoint too slow: {avg_response_time:.3f}s average"
    
    def test_job_listing_performance_with_many_jobs(self, client):
        """Test job listing performance with many jobs."""
        # Create multiple jobs
        job_ids = []
        for i in range(10):
            evaluation_request = {
                "model": {"provider": "local", "name": f"job-perf-{i}"},
                "benchmarks": ["truthfulqa"],
                "config": {"temperature": 0.0, "num_questions": 1}
            }
            response = client.post("/api/v1/evaluate", json=evaluation_request)
            job_ids.append(response.json()["job_id"])
        
        # Time job listing
        start_time = time.time()
        response = client.get("/api/v1/jobs")
        end_time = time.time()
        
        listing_time = end_time - start_time
        
        assert response.status_code == 200
        data = response.json()
        jobs = data["jobs"]
        
        # Should list all jobs
        assert len(jobs) >= 10
        
        # Should be reasonably fast even with many jobs
        assert listing_time < 1.0, f"Job listing too slow: {listing_time:.3f}s with {len(jobs)} jobs"


@pytest.mark.performance  
class TestMemoryAndResourceUsage:
    """Test memory usage and resource management."""
    
    def test_results_memory_efficiency(self):
        """Test memory efficiency of results storage."""
        import sys
        from agi_eval_sandbox.core.results import Results, BenchmarkResult, EvaluationResult
        from agi_eval_sandbox.core.benchmarks import Score
        
        # Create a large result set
        results = Results()
        
        # Measure memory before adding results
        initial_size = sys.getsizeof(results)
        
        # Add many evaluation results
        for benchmark_num in range(10):
            eval_results = []
            
            for i in range(100):
                score = Score(value=0.7, passed=True)
                eval_result = EvaluationResult(
                    question_id=f"mem-test-{benchmark_num}-{i}",
                    question_prompt=f"Memory test question {i}",
                    model_response=f"Response {i}",
                    score=score,
                    benchmark_name=f"memory_benchmark_{benchmark_num}"
                )
                eval_results.append(eval_result)
            
            benchmark_result = BenchmarkResult(
                benchmark_name=f"memory_benchmark_{benchmark_num}",
                model_name="memory_test_model",
                model_provider="local",
                results=eval_results
            )
            
            results.add_benchmark_result(benchmark_result)
        
        # Measure final memory usage
        final_size = sys.getsizeof(results)
        
        # Check that memory usage is reasonable
        # Should have 10 benchmarks * 100 questions = 1000 total results
        total_questions = sum(len(br.results) for br in results.benchmark_results)
        assert total_questions == 1000
        
        # Memory per question should be reasonable
        memory_per_question = (final_size - initial_size) / total_questions
        
        # Should use less than 1KB per question result on average
        assert memory_per_question < 1024, f"Memory usage too high: {memory_per_question:.1f} bytes per question"
    
    @pytest.mark.asyncio
    async def test_evaluation_resource_cleanup(self):
        """Test that evaluations properly clean up resources."""
        import gc
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run several evaluations
        for i in range(5):
            model = create_mock_model(f"cleanup-test-{i}", simulate_delay=0.001)
            suite = EvalSuite()
            
            # Create temporary benchmark
            questions = [
                Question(
                    id=f"cleanup-q{j}",
                    prompt=f"Cleanup test question {j}",
                    correct_answer=f"Answer {j}",
                    question_type=QuestionType.SHORT_ANSWER
                )
                for j in range(20)
            ]
            
            benchmark = CustomBenchmark(f"cleanup-test-{i}", questions)
            suite.register_benchmark(benchmark)
            
            # Run evaluation
            results = await suite.evaluate(
                model=model,
                benchmarks=[f"cleanup-test-{i}"],
                save_results=False,
                parallel=True
            )
            
            # Explicitly delete large objects
            del results
            del benchmark
            del questions
            del model
            del suite
            
            # Force garbage collection
            gc.collect()
        
        # Check final memory
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        memory_increase_mb = memory_increase / (1024 * 1024)
        
        # Memory increase should be minimal after cleanup
        assert memory_increase_mb < 100, f"Memory leak detected: {memory_increase_mb:.1f}MB increase"
    
    def test_large_benchmark_loading_memory(self):
        """Test memory usage when loading large benchmarks."""
        import sys
        
        # Create a large custom benchmark
        large_questions = [
            Question(
                id=f"large-q{i}",
                prompt=f"Large benchmark question {i}: " + "padding " * 50,
                correct_answer=f"Answer {i}",
                question_type=QuestionType.SHORT_ANSWER,
                metadata={"large_data": list(range(100))}  # Add some metadata
            )
            for i in range(1000)
        ]
        
        # Measure memory usage
        benchmark = CustomBenchmark("large-benchmark", large_questions)
        
        # Get size of benchmark object
        benchmark_size = sys.getsizeof(benchmark)
        questions_size = sum(sys.getsizeof(q) for q in benchmark.get_questions())
        
        total_size_mb = (benchmark_size + questions_size) / (1024 * 1024)
        
        # Should use reasonable amount of memory
        assert total_size_mb < 50, f"Large benchmark uses too much memory: {total_size_mb:.1f}MB"
        
        # Memory per question should be reasonable
        memory_per_question = questions_size / len(large_questions)
        assert memory_per_question < 5000, f"Memory per question too high: {memory_per_question:.1f} bytes"
    
    @pytest.mark.asyncio
    async def test_concurrent_evaluation_memory_usage(self):
        """Test memory usage during concurrent evaluations."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Create multiple models and benchmarks for concurrent evaluation
        models = [create_mock_model(f"concurrent-mem-{i}", simulate_delay=0.01) for i in range(3)]
        
        suite = EvalSuite()
        
        # Create benchmarks
        for i in range(3):
            questions = [
                Question(
                    id=f"concurrent-mem-{i}-q{j}",
                    prompt=f"Concurrent memory test {i}-{j}",
                    correct_answer=f"Answer {j}",
                    question_type=QuestionType.SHORT_ANSWER
                )
                for j in range(50)
            ]
            
            benchmark = CustomBenchmark(f"concurrent-mem-{i}", questions)
            suite.register_benchmark(benchmark)
        
        # Run concurrent evaluations
        async def run_evaluation(model_idx):
            return await suite.evaluate(
                model=models[model_idx],
                benchmarks=[f"concurrent-mem-{model_idx}"],
                save_results=False,
                parallel=True
            )
        
        # Monitor peak memory during concurrent execution
        peak_memory = initial_memory
        
        tasks = [run_evaluation(i) for i in range(3)]
        results = await asyncio.gather(*tasks)
        
        # Update peak memory
        current_memory = process.memory_info().rss
        peak_memory = max(peak_memory, current_memory)
        
        memory_increase = peak_memory - initial_memory
        memory_increase_mb = memory_increase / (1024 * 1024)
        
        # Concurrent evaluations should not use excessive memory
        assert memory_increase_mb < 200, f"Concurrent evaluations use too much memory: {memory_increase_mb:.1f}MB"
        
        # All evaluations should complete successfully
        assert all(r.summary()["total_questions"] == 50 for r in results)


@pytest.fixture
def performance_threshold():
    """Performance thresholds for testing."""
    return {
        "api_response_time": 1.0,  # seconds
        "evaluation_time": 30.0,  # seconds
        "memory_usage": 512 * 1024 * 1024,  # 512MB
        "throughput_qps": 10,  # questions per second
    }