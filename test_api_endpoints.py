#!/usr/bin/env python3
"""
Test API endpoints for the AGI Evaluation Sandbox.
Tests both evaluation and context compression endpoints.
"""

import asyncio
import sys
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    import httpx
except ImportError:
    print("❌ httpx not installed. Please install with: pip install httpx")
    sys.exit(1)


async def test_api_health(client, base_url):
    """Test basic API health endpoints."""
    print("🏥 Testing API Health...")
    
    try:
        # Test root endpoint
        response = await client.get(f"{base_url}/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "AGI Evaluation Sandbox API"
        print("✅ Root endpoint working")
        
        # Test health endpoint
        response = await client.get(f"{base_url}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        print("✅ Health endpoint working")
        
        # Test API v1 endpoint
        response = await client.get(f"{base_url}/api/v1")
        assert response.status_code == 200
        data = response.json()
        assert "endpoints" in data
        print("✅ API v1 endpoint working")
        
        return True
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


async def test_compression_endpoints(client, base_url):
    """Test context compression endpoints."""
    print("\n🗜️  Testing Context Compression Endpoints...")
    
    try:
        # Test list compression strategies
        response = await client.get(f"{base_url}/api/v1/compress/strategies")
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        strategies = data["strategies"]
        assert len(strategies) > 0
        print(f"✅ Found {len(strategies)} compression strategies")
        
        # Test compression job submission
        test_text = """
        This is a test document for context compression.
        It contains multiple sentences with different importance levels.
        Some sentences are more critical than others for understanding.
        The compression algorithm should identify key information.
        Less important details can be removed to save space.
        The goal is to maintain meaning while reducing length.
        This serves as a good test case for compression algorithms.
        """
        
        compression_data = {
            "text": test_text,
            "strategy": "extractive_summarization",
            "target_ratio": 0.5,
            "preserve_structure": True
        }
        
        response = await client.post(
            f"{base_url}/api/v1/compress",
            params=compression_data
        )
        assert response.status_code == 200
        data = response.json()
        job_id = data["job_id"]
        assert data["status"] == "pending"
        print(f"✅ Compression job submitted: {job_id}")
        
        # Wait for job completion
        print("⏳ Waiting for compression to complete...")
        max_attempts = 30
        for attempt in range(max_attempts):
            await asyncio.sleep(1)
            
            response = await client.get(f"{base_url}/api/v1/compress/jobs/{job_id}")
            assert response.status_code == 200
            job_status = response.json()
            
            if job_status["status"] == "completed":
                print("✅ Compression job completed")
                break
            elif job_status["status"] == "failed":
                print(f"❌ Compression job failed: {job_status.get('error')}")
                return False
            
            if attempt == max_attempts - 1:
                print("⏰ Compression job timed out")
                return False
        
        # Get compression results
        response = await client.get(f"{base_url}/api/v1/compress/jobs/{job_id}/results")
        assert response.status_code == 200
        results = response.json()
        assert "compressed_text" in results
        assert "metrics" in results
        
        metrics = results["metrics"]
        print(f"✅ Compression successful:")
        print(f"   Original tokens: {metrics['original_tokens']}")
        print(f"   Compressed tokens: {metrics['compressed_tokens']}")
        print(f"   Compression ratio: {metrics['compression_ratio']:.3f}")
        
        # Test benchmark endpoint
        benchmark_data = {
            "text": test_text[:500],  # Shorter text for benchmark
            "target_length": None
        }
        
        response = await client.post(
            f"{base_url}/api/v1/compress/benchmark",
            params=benchmark_data
        )
        assert response.status_code == 200
        data = response.json()
        benchmark_job_id = data["job_id"]
        print(f"✅ Benchmark job submitted: {benchmark_job_id}")
        
        # Wait for benchmark completion
        print("⏳ Waiting for benchmark to complete...")
        for attempt in range(20):
            await asyncio.sleep(2)
            
            response = await client.get(f"{base_url}/api/v1/compress/jobs/{benchmark_job_id}")
            if response.status_code == 200:
                job_status = response.json()
                if job_status["status"] == "completed":
                    print("✅ Benchmark job completed")
                    break
            
            if attempt == 19:
                print("⏰ Benchmark job timed out")
        
        # Test list compression jobs
        response = await client.get(f"{base_url}/api/v1/compress/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "compression_jobs" in data
        jobs = data["compression_jobs"]
        assert len(jobs) >= 2  # At least the two jobs we created
        print(f"✅ Found {len(jobs)} compression jobs")
        
        return True
        
    except Exception as e:
        print(f"❌ Compression endpoint test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_benchmark_endpoints(client, base_url):
    """Test benchmark-related endpoints."""
    print("\n📊 Testing Benchmark Endpoints...")
    
    try:
        # Test list benchmarks
        response = await client.get(f"{base_url}/api/v1/benchmarks")
        assert response.status_code == 200
        benchmarks = response.json()
        assert isinstance(benchmarks, list)
        print(f"✅ Found {len(benchmarks)} benchmarks")
        
        # Test get benchmark details (if any benchmarks exist)
        if benchmarks:
            benchmark_name = benchmarks[0]["name"]
            response = await client.get(f"{base_url}/api/v1/benchmarks/{benchmark_name}")
            assert response.status_code == 200
            details = response.json()
            assert "name" in details
            assert "total_questions" in details
            print(f"✅ Got details for benchmark: {benchmark_name}")
        
        # Test custom benchmarks
        response = await client.get(f"{base_url}/api/v1/benchmarks/custom")
        assert response.status_code == 200
        data = response.json()
        assert "custom_benchmarks" in data
        print("✅ Custom benchmarks endpoint working")
        
        return True
        
    except Exception as e:
        print(f"❌ Benchmark endpoint test failed: {e}")
        return False


async def test_stats_endpoint(client, base_url):
    """Test statistics endpoint."""
    print("\n📈 Testing Stats Endpoint...")
    
    try:
        response = await client.get(f"{base_url}/api/v1/stats")
        assert response.status_code == 200
        stats = response.json()
        
        expected_fields = [
            "total_jobs", "completed_jobs", "running_jobs", "failed_jobs",
            "available_benchmarks", "uptime"
        ]
        
        for field in expected_fields:
            assert field in stats, f"Missing field: {field}"
        
        print("✅ Stats endpoint working")
        print(f"   Total jobs: {stats['total_jobs']}")
        print(f"   Available benchmarks: {stats['available_benchmarks']}")
        
        # Check for compression-specific stats
        if "total_compression_jobs" in stats:
            print(f"   Compression jobs: {stats['total_compression_jobs']}")
            print(f"   Compression strategies: {stats.get('available_compression_strategies', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Stats endpoint test failed: {e}")
        return False


async def test_error_handling(client, base_url):
    """Test API error handling."""
    print("\n🚨 Testing Error Handling...")
    
    try:
        # Test invalid compression strategy
        response = await client.post(
            f"{base_url}/api/v1/compress",
            params={
                "text": "Test text",
                "strategy": "invalid_strategy"
            }
        )
        assert response.status_code == 400
        error = response.json()
        assert "detail" in error
        print("✅ Invalid strategy error handled correctly")
        
        # Test empty text compression
        response = await client.post(
            f"{base_url}/api/v1/compress",
            params={
                "text": "",
                "strategy": "extractive_summarization"
            }
        )
        assert response.status_code == 400
        print("✅ Empty text error handled correctly")
        
        # Test invalid job ID
        response = await client.get(f"{base_url}/api/v1/compress/jobs/invalid-job-id")
        assert response.status_code == 404
        print("✅ Invalid job ID error handled correctly")
        
        # Test invalid benchmark name
        response = await client.get(f"{base_url}/api/v1/benchmarks/nonexistent-benchmark")
        assert response.status_code == 404
        print("✅ Invalid benchmark error handled correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


async def run_api_tests(api_url="http://localhost:8080"):
    """Run comprehensive API tests."""
    print("🚀 Starting API Endpoint Tests")
    print("="*50)
    print(f"API URL: {api_url}")
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test if API is running
        try:
            response = await client.get(f"{api_url}/health")
            if response.status_code != 200:
                raise Exception("API not responding")
        except Exception as e:
            print(f"❌ API not accessible at {api_url}")
            print("Make sure the API server is running with: python -m agi_eval_sandbox.api.main")
            return False
        
        tests = [
            test_api_health,
            test_compression_endpoints,
            test_benchmark_endpoints,
            test_stats_endpoint,
            test_error_handling
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_func in tests:
            try:
                success = await test_func(client, api_url)
                if success:
                    passed_tests += 1
            except Exception as e:
                print(f"❌ Test {test_func.__name__} failed with exception: {e}")
                import traceback
                traceback.print_exc()
        
        print(f"\n{'='*50}")
        print("🏁 API TEST RESULTS")
        print(f"{'='*50}")
        print(f"✅ Passed: {passed_tests}/{total_tests} tests")
        print(f"📊 Success rate: {(passed_tests/total_tests*100):.1f}%")
        
        if passed_tests == total_tests:
            print("🎉 All API tests passed!")
            return True
        else:
            print("⚠️  Some API tests failed.")
            return False


async def main():
    """Main test function."""
    print("AGI Evaluation Sandbox API Test Suite")
    print("=====================================")
    
    # You can change this URL to test against a different API instance
    api_url = "http://localhost:8080"
    
    success = await run_api_tests(api_url)
    
    if success:
        print("\n✅ All API tests completed successfully!")
        return 0
    else:
        print("\n❌ Some API tests failed!")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)