"""
Integration tests for the API endpoints.
"""

import pytest
import asyncio
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from agi_eval_sandbox.api.main import app


@pytest.fixture
def client():
    """Test client for API endpoints."""
    return TestClient(app)


class TestHealthEndpoints:
    """Test health and status endpoints."""
    
    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert data["message"] == "AGI Evaluation Sandbox API"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "active_jobs" in data


class TestBenchmarkEndpoints:
    """Test benchmark-related endpoints."""
    
    def test_list_benchmarks(self, client):
        """Test listing available benchmarks."""
        response = client.get("/benchmarks")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        assert len(data) > 0
        
        # Check benchmark structure
        benchmark = data[0]
        required_fields = ["name", "version", "total_questions", "categories", "description"]
        for field in required_fields:
            assert field in benchmark
    
    def test_get_benchmark_details(self, client):
        """Test getting specific benchmark details."""
        # First get list of benchmarks
        response = client.get("/benchmarks")
        benchmarks = response.json()
        
        if benchmarks:
            benchmark_name = benchmarks[0]["name"]
            
            # Get details for first benchmark
            response = client.get(f"/benchmarks/{benchmark_name}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["name"] == benchmark_name
            assert "total_questions" in data
            assert "question_types" in data
            assert "sample_questions" in data
    
    def test_get_nonexistent_benchmark(self, client):
        """Test getting details for non-existent benchmark."""
        response = client.get("/benchmarks/nonexistent")
        assert response.status_code == 404


class TestEvaluationEndpoints:
    """Test evaluation endpoints."""
    
    def test_start_evaluation_missing_data(self, client):
        """Test starting evaluation with missing data."""
        response = client.post("/evaluate", json={})
        assert response.status_code == 422  # Validation error
    
    def test_start_evaluation_valid_data(self, client):
        """Test starting evaluation with valid data."""
        evaluation_request = {
            "model": {
                "provider": "local",
                "name": "test-model"
            },
            "benchmarks": ["truthfulqa"],
            "config": {
                "temperature": 0.0,
                "max_tokens": 100,
                "num_questions": 1
            }
        }
        
        response = client.post("/evaluate", json=evaluation_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["message"] == "Evaluation job started"


class TestJobEndpoints:
    """Test job management endpoints."""
    
    def test_get_nonexistent_job(self, client):
        """Test getting status of non-existent job."""
        fake_job_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.get(f"/jobs/{fake_job_id}")
        assert response.status_code == 404
    
    def test_list_jobs_empty(self, client):
        """Test listing jobs when none exist."""
        response = client.get("/jobs")
        assert response.status_code == 200
        
        data = response.json()
        assert "jobs" in data
        assert isinstance(data["jobs"], list)
    
    def test_cancel_nonexistent_job(self, client):
        """Test cancelling non-existent job."""
        fake_job_id = "550e8400-e29b-41d4-a716-446655440000"
        response = client.delete(f"/jobs/{fake_job_id}")
        assert response.status_code == 404


class TestLeaderboardEndpoints:
    """Test leaderboard endpoints.""" 
    
    def test_get_leaderboard_empty(self, client):
        """Test getting leaderboard when no results exist."""
        response = client.get("/leaderboard")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)
        # Should be empty initially
        assert len(data) == 0
    
    def test_get_leaderboard_with_filters(self, client):
        """Test getting leaderboard with query parameters."""
        response = client.get("/leaderboard?benchmark=truthfulqa&limit=10")
        assert response.status_code == 200
        
        data = response.json()
        assert isinstance(data, list)


class TestStatsEndpoint:
    """Test statistics endpoint."""
    
    def test_get_stats(self, client):
        """Test getting API statistics."""
        response = client.get("/stats")
        assert response.status_code == 200
        
        data = response.json()
        required_fields = [
            "total_jobs", "completed_jobs", "running_jobs", 
            "failed_jobs", "available_benchmarks", "uptime"
        ]
        
        for field in required_fields:
            assert field in data
            assert isinstance(data[field], (int, str))


class TestComparisonEndpoints:
    """Test model comparison endpoints."""
    
    def test_compare_models_missing_data(self, client):
        """Test model comparison with missing data."""
        response = client.post("/compare", json={})
        assert response.status_code == 400
    
    def test_compare_models_insufficient_models(self, client):
        """Test model comparison with only one model."""
        comparison_request = {
            "models": [{
                "provider": "local",
                "name": "test-model"
            }],
            "benchmarks": ["truthfulqa"],
            "config": {
                "temperature": 0.0,
                "max_tokens": 100
            }
        }
        
        response = client.post("/compare", json=comparison_request)
        assert response.status_code == 400
        assert "At least 2 models required" in response.json()["detail"]
    
    def test_compare_models_valid_data(self, client):
        """Test model comparison with valid data."""
        comparison_request = {
            "models": [
                {
                    "provider": "local",
                    "name": "test-model-1"
                },
                {
                    "provider": "local", 
                    "name": "test-model-2"
                }
            ],
            "benchmarks": ["truthfulqa"],
            "config": {
                "temperature": 0.0,
                "max_tokens": 100,
                "num_questions": 1
            }
        }
        
        response = client.post("/compare", json=comparison_request)
        assert response.status_code == 200
        
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["message"] == "Model comparison started"


@pytest.mark.integration
class TestEndToEndFlow:
    """Test complete end-to-end flows."""
    
    def test_evaluation_flow(self, client):
        """Test complete evaluation flow."""
        # 1. Check initial stats
        stats_response = client.get("/stats")
        initial_stats = stats_response.json()
        initial_jobs = initial_stats["total_jobs"]
        
        # 2. Start evaluation
        evaluation_request = {
            "model": {
                "provider": "local",
                "name": "e2e-test-model"
            },
            "benchmarks": ["truthfulqa"],
            "config": {
                "temperature": 0.0,
                "max_tokens": 50,
                "num_questions": 1
            }
        }
        
        eval_response = client.post("/evaluate", json=evaluation_request)
        assert eval_response.status_code == 200
        
        job_data = eval_response.json()
        job_id = job_data["job_id"]
        
        # 3. Check job appears in jobs list
        jobs_response = client.get("/jobs")
        jobs_data = jobs_response.json()
        job_ids = [job["job_id"] for job in jobs_data["jobs"]]
        assert job_id in job_ids
        
        # 4. Check stats updated
        stats_response = client.get("/stats")
        updated_stats = stats_response.json()
        assert updated_stats["total_jobs"] == initial_jobs + 1
        
        # 5. Check job status
        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert status_data["job_id"] == job_id
        assert status_data["status"] in ["pending", "running", "completed", "failed"]
    
    def test_comparison_flow(self, client):
        """Test complete model comparison flow."""
        # Start comparison
        comparison_request = {
            "models": [
                {"provider": "local", "name": "model-a"},
                {"provider": "local", "name": "model-b"}
            ],
            "benchmarks": ["truthfulqa"],
            "config": {
                "temperature": 0.0,
                "max_tokens": 50,
                "num_questions": 1
            }
        }
        
        compare_response = client.post("/compare", json=comparison_request)
        assert compare_response.status_code == 200
        
        job_data = compare_response.json()
        job_id = job_data["job_id"]
        
        # Check job status
        status_response = client.get(f"/jobs/{job_id}")
        assert status_response.status_code == 200
        
        status_data = status_response.json()
        assert status_data["job_id"] == job_id