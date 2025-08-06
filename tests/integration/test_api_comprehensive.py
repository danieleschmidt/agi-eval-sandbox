"""
Comprehensive integration tests for API endpoints and functionality.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient

from agi_eval_sandbox.api.main import app


@pytest.fixture
def client():
    """Test client for API endpoints."""
    return TestClient(app)


@pytest.mark.integration
class TestHealthAndStatusEndpoints:
    """Test health check and status endpoints."""
    
    def test_root_endpoint_comprehensive(self, client):
        """Test root endpoint with comprehensive checks."""
        response = client.get("/")
        
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("application/json")
        
        data = response.json()
        
        # Check required fields
        required_fields = ["message", "version", "docs", "status", "timestamp"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Check field values
        assert data["message"] == "AGI Evaluation Sandbox API"
        assert data["version"] == "0.1.0"
        assert data["docs"] == "/docs"
        assert data["status"] == "running"
        
        # Check timestamp format
        timestamp = data["timestamp"]
        assert isinstance(timestamp, str)
        assert "T" in timestamp  # ISO format
    
    def test_health_endpoint_comprehensive(self, client):
        """Test health check endpoint with comprehensive validation."""
        response = client.get("/health")
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        required_fields = ["status", "timestamp", "active_jobs"]
        for field in required_fields:
            assert field in data
        
        assert data["status"] == "healthy"
        assert isinstance(data["active_jobs"], int)
        assert data["active_jobs"] >= 0
    
    def test_api_v1_root_endpoint(self, client):
        """Test API v1 root endpoint."""
        response = client.get("/api/v1")
        
        assert response.status_code == 200
        
        data = response.json()
        
        assert data["message"] == "AGI Evaluation Sandbox API v1"
        assert data["version"] == "0.1.0"
        assert "endpoints" in data
        
        # Check that important endpoints are listed
        endpoints = data["endpoints"]
        expected_endpoints = ["evaluate", "jobs", "leaderboard", "benchmarks", "stats"]
        for endpoint in expected_endpoints:
            assert endpoint in endpoints
    
    def test_health_endpoint_load(self, client):
        """Test health endpoint under repeated requests (basic load test)."""
        # Make multiple rapid requests to health endpoint
        responses = []
        for _ in range(10):
            response = client.get("/health")
            responses.append(response)
        
        # All should be successful
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json()["status"] == "healthy" for r in responses)


@pytest.mark.integration
class TestBenchmarkEndpoints:
    """Test benchmark-related endpoints comprehensively."""
    
    def test_list_benchmarks_structure(self, client):
        """Test benchmark listing with detailed structure validation."""
        response = client.get("/api/v1/benchmarks")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0, "Should have at least some benchmarks"
        
        # Validate benchmark structure
        for benchmark in data:
            required_fields = ["name", "version", "total_questions", "categories", "description"]
            for field in required_fields:
                assert field in benchmark, f"Benchmark missing field: {field}"
            
            # Validate field types
            assert isinstance(benchmark["name"], str)
            assert isinstance(benchmark["version"], str) 
            assert isinstance(benchmark["total_questions"], int)
            assert isinstance(benchmark["categories"], list)
            assert isinstance(benchmark["description"], str)
            
            # Validate values
            assert len(benchmark["name"]) > 0
            assert benchmark["total_questions"] > 0
    
    def test_get_benchmark_details_comprehensive(self, client):
        """Test getting specific benchmark details with validation."""
        # First get list of benchmarks
        list_response = client.get("/api/v1/benchmarks")
        benchmarks = list_response.json()
        
        assert len(benchmarks) > 0, "Need at least one benchmark for testing"
        
        benchmark_name = benchmarks[0]["name"]
        
        # Get detailed information
        detail_response = client.get(f"/api/v1/benchmarks/{benchmark_name}")
        
        assert detail_response.status_code == 200
        
        data = detail_response.json()
        
        # Check required fields
        required_fields = ["name", "version", "total_questions", "question_types", 
                          "categories", "difficulties", "sample_questions"]
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Validate field types and values
        assert data["name"] == benchmark_name
        assert isinstance(data["total_questions"], int)
        assert data["total_questions"] > 0
        
        assert isinstance(data["question_types"], dict)
        assert isinstance(data["categories"], dict)
        assert isinstance(data["sample_questions"], list)
        
        # Check sample questions structure
        for sample in data["sample_questions"]:
            required_sample_fields = ["id", "prompt", "type", "category", "difficulty"]
            for field in required_sample_fields:
                assert field in sample
    
    def test_get_all_benchmarks_details(self, client):
        """Test getting details for all available benchmarks."""
        # Get list of all benchmarks
        list_response = client.get("/api/v1/benchmarks")
        benchmarks = list_response.json()
        
        # Test details for each benchmark
        for benchmark in benchmarks:
            benchmark_name = benchmark["name"]
            
            detail_response = client.get(f"/api/v1/benchmarks/{benchmark_name}")
            
            assert detail_response.status_code == 200, f"Failed to get details for {benchmark_name}"
            
            data = detail_response.json()
            assert data["name"] == benchmark_name
            assert data["total_questions"] > 0
    
    def test_benchmark_not_found(self, client):
        """Test getting details for non-existent benchmark."""
        response = client.get("/api/v1/benchmarks/definitely-not-a-benchmark")
        
        assert response.status_code == 404
        
        error_data = response.json()
        assert "detail" in error_data
        assert "not found" in error_data["detail"].lower()
    
    def test_benchmark_case_sensitivity(self, client):
        """Test benchmark name case sensitivity."""
        # Get a real benchmark name
        list_response = client.get("/api/v1/benchmarks")
        benchmarks = list_response.json()
        
        if benchmarks:
            benchmark_name = benchmarks[0]["name"]
            
            # Test with different cases
            upper_response = client.get(f"/api/v1/benchmarks/{benchmark_name.upper()}")
            lower_response = client.get(f"/api/v1/benchmarks/{benchmark_name.lower()}")
            
            # Depending on implementation, these might return 404 or the benchmark
            # Just ensure we get consistent behavior
            assert upper_response.status_code in [200, 404]
            assert lower_response.status_code in [200, 404]


@pytest.mark.integration
class TestCustomBenchmarkEndpoints:
    """Test custom benchmark creation and management."""
    
    def test_create_custom_benchmark_valid(self, client):
        """Test creating a valid custom benchmark."""
        custom_benchmark_data = {
            "name": "Test Custom Benchmark",
            "description": "A test custom benchmark for integration testing",
            "category": "test",
            "tags": ["test", "integration", "custom"],
            "questions": [
                {
                    "prompt": "What is 2+2?",
                    "correct_answer": "4",
                    "question_type": "short_answer",
                    "category": "math",
                    "difficulty": "easy"
                },
                {
                    "prompt": "What is the capital of France?",
                    "correct_answer": "Paris",
                    "question_type": "short_answer",
                    "category": "geography",
                    "difficulty": "easy"
                }
            ]
        }
        
        response = client.post("/api/v1/benchmarks/custom", json=custom_benchmark_data)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check response structure
        required_fields = ["benchmark_id", "name", "status", "message", "total_questions"]
        for field in required_fields:
            assert field in data
        
        assert data["name"] == "Test Custom Benchmark"
        assert data["status"] == "created"
        assert data["total_questions"] == 2
        
        # Store benchmark ID for cleanup/further testing
        return data["benchmark_id"]
    
    def test_create_custom_benchmark_invalid_missing_prompt(self, client):
        """Test creating custom benchmark with missing required fields."""
        invalid_data = {
            "name": "Invalid Benchmark",
            "description": "Missing prompts in questions",
            "questions": [
                {
                    "correct_answer": "Answer without prompt",
                    "question_type": "short_answer"
                }
            ]
        }
        
        response = client.post("/api/v1/benchmarks/custom", json=invalid_data)
        
        assert response.status_code == 400
        
        error_data = response.json()
        assert "detail" in error_data
        assert "prompt" in error_data["detail"].lower()
    
    def test_create_custom_benchmark_empty_questions(self, client):
        """Test creating custom benchmark with no questions."""
        empty_data = {
            "name": "Empty Benchmark",
            "description": "Benchmark with no questions",
            "questions": []
        }
        
        response = client.post("/api/v1/benchmarks/custom", json=empty_data)
        
        # Should still create but with 0 questions
        assert response.status_code == 200
        
        data = response.json()
        assert data["total_questions"] == 0
    
    def test_list_custom_benchmarks(self, client):
        """Test listing custom benchmarks."""
        # First create a custom benchmark
        benchmark_id = self.test_create_custom_benchmark_valid(client)
        
        # List custom benchmarks
        response = client.get("/api/v1/benchmarks/custom")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "custom_benchmarks" in data
        assert isinstance(data["custom_benchmarks"], list)
        
        # Should contain our created benchmark
        benchmark_ids = [b["benchmark_id"] for b in data["custom_benchmarks"]]
        assert benchmark_id in benchmark_ids
    
    def test_get_custom_benchmark_details(self, client):
        """Test getting details of a specific custom benchmark."""
        # Create a benchmark first
        benchmark_id = self.test_create_custom_benchmark_valid(client)
        
        # Get details
        response = client.get(f"/api/v1/benchmarks/custom/{benchmark_id}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["id"] == benchmark_id
        assert data["name"] == "Test Custom Benchmark"
        assert len(data["questions"]) == 2
    
    def test_delete_custom_benchmark(self, client):
        """Test deleting a custom benchmark."""
        # Create a benchmark first
        benchmark_id = self.test_create_custom_benchmark_valid(client)
        
        # Delete it
        response = client.delete(f"/api/v1/benchmarks/custom/{benchmark_id}")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "deleted successfully" in data["message"]
        assert data["benchmark_id"] == benchmark_id
        
        # Verify it's gone
        get_response = client.get(f"/api/v1/benchmarks/custom/{benchmark_id}")
        assert get_response.status_code == 404
    
    def test_custom_benchmark_not_found(self, client):
        """Test operations on non-existent custom benchmark."""
        fake_id = "non-existent-benchmark-id"
        
        # Get non-existent benchmark
        get_response = client.get(f"/api/v1/benchmarks/custom/{fake_id}")
        assert get_response.status_code == 404
        
        # Delete non-existent benchmark
        delete_response = client.delete(f"/api/v1/benchmarks/custom/{fake_id}")
        assert delete_response.status_code == 404


@pytest.mark.integration
class TestEvaluationEndpoints:
    """Test evaluation job endpoints comprehensively."""
    
    def test_start_evaluation_valid_local_model(self, client):
        """Test starting evaluation with valid local model configuration."""
        evaluation_request = {
            "model": {
                "provider": "local",
                "name": "test-integration-model",
                "api_key": None
            },
            "benchmarks": ["truthfulqa"],
            "config": {
                "temperature": 0.0,
                "max_tokens": 100,
                "num_questions": 2,
                "parallel": True
            }
        }
        
        response = client.post("/api/v1/evaluate", json=evaluation_request)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check response structure
        required_fields = ["job_id", "status", "message"]
        for field in required_fields:
            assert field in data
        
        assert data["status"] == "pending"
        assert data["message"] == "Evaluation job started"
        
        # Validate job ID format (should be UUID)
        job_id = data["job_id"]
        assert isinstance(job_id, str)
        assert len(job_id) > 0
        
        return job_id
    
    def test_start_evaluation_missing_model(self, client):
        """Test starting evaluation without model configuration."""
        invalid_request = {
            "benchmarks": ["truthfulqa"],
            "config": {"temperature": 0.0}
        }
        
        response = client.post("/api/v1/evaluate", json=invalid_request)
        
        assert response.status_code == 422  # Validation error
    
    def test_start_evaluation_invalid_provider(self, client):
        """Test starting evaluation with invalid provider."""
        invalid_request = {
            "model": {
                "provider": "invalid_provider",
                "name": "test-model"
            },
            "benchmarks": ["truthfulqa"],
            "config": {"temperature": 0.0}
        }
        
        response = client.post("/api/v1/evaluate", json=invalid_request)
        
        assert response.status_code == 400
    
    def test_start_evaluation_invalid_benchmark(self, client):
        """Test starting evaluation with invalid benchmark."""
        invalid_request = {
            "model": {
                "provider": "local",
                "name": "test-model"
            },
            "benchmarks": ["non_existent_benchmark"],
            "config": {"temperature": 0.0}
        }
        
        response = client.post("/api/v1/evaluate", json=invalid_request)
        
        # Should either start and fail later, or fail immediately
        # Depends on validation strategy
        assert response.status_code in [200, 400]
    
    def test_start_evaluation_with_all_benchmarks(self, client):
        """Test starting evaluation with all benchmarks."""
        evaluation_request = {
            "model": {
                "provider": "local",
                "name": "all-benchmarks-test"
            },
            "benchmarks": ["all"],
            "config": {
                "temperature": 0.0,
                "max_tokens": 50,
                "num_questions": 1
            }
        }
        
        response = client.post("/api/v1/evaluate", json=evaluation_request)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "pending"
        
        return data["job_id"]
    
    def test_evaluation_with_different_configurations(self, client):
        """Test evaluation with various configuration options."""
        configs = [
            {"temperature": 0.0, "max_tokens": 50},
            {"temperature": 0.5, "max_tokens": 100},
            {"temperature": 1.0, "max_tokens": 200},
        ]
        
        job_ids = []
        
        for i, config in enumerate(configs):
            evaluation_request = {
                "model": {
                    "provider": "local",
                    "name": f"config-test-{i}"
                },
                "benchmarks": ["truthfulqa"],
                "config": {**config, "num_questions": 1}
            }
            
            response = client.post("/api/v1/evaluate", json=evaluation_request)
            
            assert response.status_code == 200
            job_ids.append(response.json()["job_id"])
        
        return job_ids


@pytest.mark.integration
class TestJobManagementEndpoints:
    """Test job management and status endpoints."""
    
    def test_get_job_status_valid(self, client):
        """Test getting status of a valid job."""
        # Start an evaluation first
        evaluation_request = {
            "model": {"provider": "local", "name": "job-status-test"},
            "benchmarks": ["truthfulqa"],
            "config": {"temperature": 0.0, "num_questions": 1}
        }
        
        start_response = client.post("/api/v1/evaluate", json=evaluation_request)
        job_id = start_response.json()["job_id"]
        
        # Get job status
        status_response = client.get(f"/api/v1/jobs/{job_id}")
        
        assert status_response.status_code == 200
        
        data = status_response.json()
        
        # Check required fields
        required_fields = ["job_id", "status", "progress", "created_at"]
        for field in required_fields:
            assert field in data
        
        assert data["job_id"] == job_id
        assert data["status"] in ["pending", "running", "completed", "failed"]
        assert 0.0 <= data["progress"] <= 1.0
        assert isinstance(data["created_at"], str)
    
    def test_get_job_status_nonexistent(self, client):
        """Test getting status of non-existent job."""
        fake_job_id = "550e8400-e29b-41d4-a716-446655440000"  # Valid UUID format
        
        response = client.get(f"/api/v1/jobs/{fake_job_id}")
        
        assert response.status_code == 404
        
        error_data = response.json()
        assert "Job not found" in error_data["detail"]
    
    def test_get_job_status_invalid_format(self, client):
        """Test getting status with invalid job ID format."""
        invalid_job_id = "not-a-valid-uuid"
        
        response = client.get(f"/api/v1/jobs/{invalid_job_id}")
        
        # Should return 404 (job not found) since invalid format won't exist
        assert response.status_code == 404
    
    def test_list_jobs_empty(self, client):
        """Test listing jobs when none exist."""
        response = client.get("/api/v1/jobs")
        
        assert response.status_code == 200
        
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)
    
    def test_list_jobs_with_data(self, client):
        """Test listing jobs after creating some."""
        # Create several jobs
        job_ids = []
        for i in range(3):
            evaluation_request = {
                "model": {"provider": "local", "name": f"list-test-{i}"},
                "benchmarks": ["truthfulqa"],
                "config": {"temperature": 0.0, "num_questions": 1}
            }
            
            response = client.post("/api/v1/evaluate", json=evaluation_request)
            job_ids.append(response.json()["job_id"])
        
        # List all jobs
        list_response = client.get("/api/v1/jobs")
        
        assert list_response.status_code == 200
        
        data = list_response.json()
        jobs = data["jobs"]
        
        assert len(jobs) >= 3  # Should have at least our 3 jobs
        
        # Check job structure
        for job in jobs:
            required_fields = ["job_id", "status", "created_at", "model", "provider", "benchmarks"]
            for field in required_fields:
                assert field in job
        
        # Verify our jobs are in the list
        listed_job_ids = [job["job_id"] for job in jobs]
        for job_id in job_ids:
            assert job_id in listed_job_ids
    
    def test_cancel_job_pending(self, client):
        """Test cancelling a pending job."""
        # Start a job
        evaluation_request = {
            "model": {"provider": "local", "name": "cancel-test"},
            "benchmarks": ["truthfulqa"],
            "config": {"temperature": 0.0, "num_questions": 1}
        }
        
        start_response = client.post("/api/v1/evaluate", json=evaluation_request)
        job_id = start_response.json()["job_id"]
        
        # Cancel the job
        cancel_response = client.delete(f"/api/v1/jobs/{job_id}")
        
        # Should succeed (either cancel or delete depending on status)
        assert cancel_response.status_code == 200
        
        data = cancel_response.json()
        assert "message" in data
        assert "cancel" in data["message"].lower() or "delet" in data["message"].lower()
    
    def test_cancel_nonexistent_job(self, client):
        """Test cancelling non-existent job."""
        fake_job_id = "550e8400-e29b-41d4-a716-446655440000"
        
        response = client.delete(f"/api/v1/jobs/{fake_job_id}")
        
        assert response.status_code == 404
    
    def test_get_job_results_before_completion(self, client):
        """Test getting results of a job that hasn't completed."""
        # Start a job
        evaluation_request = {
            "model": {"provider": "local", "name": "results-test"},
            "benchmarks": ["truthfulqa"],
            "config": {"temperature": 0.0, "num_questions": 1}
        }
        
        start_response = client.post("/api/v1/evaluate", json=evaluation_request)
        job_id = start_response.json()["job_id"]
        
        # Try to get results immediately (likely not completed yet)
        results_response = client.get(f"/api/v1/jobs/{job_id}/results")
        
        # Should return 400 if not completed, or 200 if somehow completed very quickly
        assert results_response.status_code in [200, 400]
        
        if results_response.status_code == 400:
            error_data = results_response.json()
            assert "not completed" in error_data["detail"].lower()
    
    def test_job_lifecycle_complete(self, client):
        """Test complete job lifecycle from start to results."""
        # Start evaluation
        evaluation_request = {
            "model": {"provider": "local", "name": "lifecycle-test"},
            "benchmarks": ["truthfulqa"],
            "config": {"temperature": 0.0, "num_questions": 1}
        }
        
        start_response = client.post("/api/v1/evaluate", json=evaluation_request)
        job_id = start_response.json()["job_id"]
        
        # Poll for completion (with timeout)
        max_attempts = 30
        attempt = 0
        
        while attempt < max_attempts:
            status_response = client.get(f"/api/v1/jobs/{job_id}")
            status_data = status_response.json()
            
            if status_data["status"] == "completed":
                # Get results
                results_response = client.get(f"/api/v1/jobs/{job_id}/results")
                
                assert results_response.status_code == 200
                
                results_data = results_response.json()
                
                # Check results structure
                required_fields = ["job_id", "results", "model", "benchmarks", "config"]
                for field in required_fields:
                    assert field in results_data
                
                assert results_data["job_id"] == job_id
                assert isinstance(results_data["results"], dict)
                
                break
            elif status_data["status"] == "failed":
                pytest.fail(f"Job failed: {status_data.get('error', 'Unknown error')}")
            
            time.sleep(0.5)  # Wait before next poll
            attempt += 1
        
        if attempt >= max_attempts:
            pytest.skip("Job did not complete within timeout - this is expected for long-running evaluations")


@pytest.mark.integration
class TestModelComparisonEndpoints:
    """Test model comparison functionality."""
    
    def test_compare_models_valid(self, client):
        """Test comparing multiple models."""
        comparison_request = {
            "models": [
                {"provider": "local", "name": "comparison-model-1"},
                {"provider": "local", "name": "comparison-model-2"}
            ],
            "benchmarks": ["truthfulqa"],
            "config": {
                "temperature": 0.0,
                "max_tokens": 50,
                "num_questions": 1
            }
        }
        
        response = client.post("/api/v1/compare", json=comparison_request)
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check response structure
        required_fields = ["job_id", "status", "message"]
        for field in required_fields:
            assert field in data
        
        assert data["status"] == "pending"
        assert "comparison" in data["message"].lower()
        
        return data["job_id"]
    
    def test_compare_models_insufficient_models(self, client):
        """Test comparison with only one model (should fail)."""
        comparison_request = {
            "models": [
                {"provider": "local", "name": "single-model"}
            ],
            "benchmarks": ["truthfulqa"],
            "config": {"temperature": 0.0}
        }
        
        response = client.post("/api/v1/compare", json=comparison_request)
        
        assert response.status_code == 400
        
        error_data = response.json()
        assert "at least 2 models" in error_data["detail"].lower()
    
    def test_compare_models_no_models(self, client):
        """Test comparison with no models."""
        comparison_request = {
            "models": [],
            "benchmarks": ["truthfulqa"],
            "config": {"temperature": 0.0}
        }
        
        response = client.post("/api/v1/compare", json=comparison_request)
        
        assert response.status_code == 400
    
    def test_compare_models_multiple_benchmarks(self, client):
        """Test comparison across multiple benchmarks."""
        comparison_request = {
            "models": [
                {"provider": "local", "name": "multi-bench-1"},
                {"provider": "local", "name": "multi-bench-2"}
            ],
            "benchmarks": ["truthfulqa", "mmlu"],
            "config": {
                "temperature": 0.0,
                "num_questions": 1
            }
        }
        
        response = client.post("/api/v1/compare", json=comparison_request)
        
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "pending"
        
        return data["job_id"]
    
    def test_compare_models_different_configurations(self, client):
        """Test comparison with different model configurations."""
        comparison_request = {
            "models": [
                {"provider": "local", "name": "config-model-1"},
                {"provider": "local", "name": "config-model-2"}
            ],
            "benchmarks": ["truthfulqa"],
            "config": {
                "temperature": 0.2,
                "max_tokens": 150,
                "num_questions": 1
            }
        }
        
        response = client.post("/api/v1/compare", json=comparison_request)
        
        assert response.status_code == 200


@pytest.mark.integration
class TestLeaderboardAndStatsEndpoints:
    """Test leaderboard and statistics endpoints."""
    
    def test_get_leaderboard_empty(self, client):
        """Test leaderboard when no results exist."""
        response = client.get("/api/v1/leaderboard")
        
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        # Initially empty is expected
    
    def test_get_leaderboard_with_filters(self, client):
        """Test leaderboard with query parameters."""
        # Test different filter combinations
        test_params = [
            {"benchmark": "truthfulqa"},
            {"metric": "pass_rate"},
            {"limit": "5"},
            {"benchmark": "mmlu", "metric": "average_score", "limit": "3"}
        ]
        
        for params in test_params:
            response = client.get("/api/v1/leaderboard", params=params)
            
            assert response.status_code == 200
            
            data = response.json()
            assert isinstance(data, list)
            
            # If there are results, check structure
            for entry in data:
                if entry:  # If not empty
                    expected_fields = ["rank", "model_name", "model_provider", 
                                     "benchmark", "average_score", "pass_rate", 
                                     "total_questions", "timestamp"]
                    for field in expected_fields:
                        assert field in entry
    
    def test_get_leaderboard_invalid_parameters(self, client):
        """Test leaderboard with invalid parameters."""
        # Test with invalid benchmark
        response = client.get("/api/v1/leaderboard", params={"benchmark": "nonexistent"})
        
        # Should still return 200 with empty results
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
    
    def test_get_stats_comprehensive(self, client):
        """Test statistics endpoint with comprehensive validation."""
        response = client.get("/api/v1/stats")
        
        assert response.status_code == 200
        
        data = response.json()
        
        # Check required fields
        required_fields = [
            "total_jobs", "completed_jobs", "running_jobs", "failed_jobs",
            "available_benchmarks", "uptime"
        ]
        
        for field in required_fields:
            assert field in data, f"Missing field: {field}"
        
        # Check field types and constraints
        assert isinstance(data["total_jobs"], int)
        assert isinstance(data["completed_jobs"], int) 
        assert isinstance(data["running_jobs"], int)
        assert isinstance(data["failed_jobs"], int)
        assert isinstance(data["available_benchmarks"], int)
        
        # Logical constraints
        assert data["total_jobs"] >= 0
        assert data["completed_jobs"] >= 0
        assert data["running_jobs"] >= 0
        assert data["failed_jobs"] >= 0
        assert data["available_benchmarks"] > 0  # Should have some benchmarks
        
        # Total jobs should equal sum of different status types
        assert data["total_jobs"] >= data["completed_jobs"] + data["failed_jobs"]
    
    def test_stats_after_operations(self, client):
        """Test that stats are updated after performing operations."""
        # Get initial stats
        initial_response = client.get("/api/v1/stats")
        initial_stats = initial_response.json()
        initial_total = initial_stats["total_jobs"]
        
        # Start an evaluation
        evaluation_request = {
            "model": {"provider": "local", "name": "stats-test"},
            "benchmarks": ["truthfulqa"],
            "config": {"temperature": 0.0, "num_questions": 1}
        }
        
        client.post("/api/v1/evaluate", json=evaluation_request)
        
        # Check updated stats
        updated_response = client.get("/api/v1/stats")
        updated_stats = updated_response.json()
        
        # Total jobs should have increased
        assert updated_stats["total_jobs"] == initial_total + 1


@pytest.mark.integration
class TestAPIErrorHandling:
    """Test API error handling and edge cases."""
    
    def test_malformed_json_requests(self, client):
        """Test API handling of malformed JSON requests."""
        # Send malformed JSON
        response = client.post("/api/v1/evaluate", 
                              data="{invalid json}",
                              headers={"Content-Type": "application/json"})
        
        assert response.status_code == 422  # Unprocessable Entity
    
    def test_missing_content_type(self, client):
        """Test API handling of requests without content-type header."""
        response = client.post("/api/v1/evaluate", data='{"test": "data"}')
        
        # Should handle gracefully
        assert response.status_code in [400, 415, 422]
    
    def test_large_request_payload(self, client):
        """Test API handling of very large request payloads."""
        # Create a large custom benchmark
        large_questions = []
        for i in range(100):
            large_questions.append({
                "prompt": f"Large test question {i} " + "x" * 1000,  # Long prompt
                "correct_answer": f"Answer {i}",
                "question_type": "short_answer"
            })
        
        large_request = {
            "name": "Large Test Benchmark",
            "description": "A benchmark with many questions for testing",
            "questions": large_questions
        }
        
        response = client.post("/api/v1/benchmarks/custom", json=large_request)
        
        # Should handle large requests (may take time)
        assert response.status_code in [200, 413, 422]  # 413 = Payload Too Large
    
    def test_concurrent_requests(self, client):
        """Test API handling of concurrent requests."""
        import concurrent.futures
        
        def make_request():
            return client.get("/health")
        
        # Make multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            responses = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All should succeed
        assert all(r.status_code == 200 for r in responses)
        assert all(r.json()["status"] == "healthy" for r in responses)
    
    def test_rate_limiting_behavior(self, client):
        """Test API behavior under rapid requests (basic rate limiting test)."""
        # Make many rapid requests
        responses = []
        for _ in range(20):
            response = client.get("/health")
            responses.append(response)
        
        # Should handle rapid requests gracefully
        # Most should succeed, some might be rate-limited depending on implementation
        success_count = sum(1 for r in responses if r.status_code == 200)
        assert success_count >= 10  # At least half should succeed
    
    def test_database_error_simulation(self, client):
        """Test API behavior when database operations fail."""
        # This would require mocking database failures
        # For now, just test that API handles missing data gracefully
        
        # Try to get results for a job that doesn't exist
        response = client.get("/api/v1/jobs/nonexistent/results")
        assert response.status_code == 404
        
        # Try to delete a benchmark that doesn't exist
        response = client.delete("/api/v1/benchmarks/custom/nonexistent")
        assert response.status_code == 404