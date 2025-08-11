"""Simplified FastAPI application."""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import asyncio
import logging
import uuid
import time
from datetime import datetime
import json

from ..core import EvalSuite, Model
from ..config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
eval_suite = EvalSuite()
active_jobs: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("🚀 Starting AGI Evaluation Sandbox API")
    yield
    logger.info("🛑 Shutting down AGI Evaluation Sandbox API")


# Create FastAPI app
app = FastAPI(
    title="AGI Evaluation Sandbox API",
    description="Comprehensive evaluation platform for large language models",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "AGI Evaluation Sandbox API",
        "version": "0.1.0",
        "docs": "/docs",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_jobs": len(active_jobs)
    }


@app.get("/api/v1/benchmarks")
async def get_benchmarks():
    """Get list of available benchmarks."""
    benchmarks = eval_suite.list_benchmarks()
    return {
        "benchmarks": benchmarks,
        "count": len(benchmarks)
    }


@app.get("/api/v1/benchmarks/{benchmark_name}")
async def get_benchmark_info(benchmark_name: str):
    """Get information about a specific benchmark."""
    benchmark = eval_suite.get_benchmark(benchmark_name)
    if not benchmark:
        raise HTTPException(status_code=404, detail=f"Benchmark '{benchmark_name}' not found")
    
    questions = benchmark.get_questions()
    sample_questions = benchmark.get_sample_questions(3)
    
    return {
        "name": benchmark.name,
        "version": benchmark.version,
        "total_questions": len(questions),
        "sample_questions": [
            {
                "id": q.id,
                "prompt": q.prompt[:200] + "..." if len(q.prompt) > 200 else q.prompt,
                "category": q.category,
                "difficulty": q.difficulty,
                "question_type": q.question_type.value
            } for q in sample_questions
        ]
    }


@app.post("/api/v1/evaluate")
async def create_evaluation(
    request: dict,
    background_tasks: BackgroundTasks
):
    """Create a new evaluation job."""
    
    # Basic request validation
    required_fields = ["model", "provider"]
    for field in required_fields:
        if field not in request:
            raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
    
    # Create job ID
    job_id = str(uuid.uuid4())
    
    # Store job info
    active_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "request": request,
        "result": None,
        "error": None
    }
    
    # Start evaluation in background
    background_tasks.add_task(run_evaluation, job_id, request)
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Evaluation job created successfully"
    }


@app.get("/api/v1/jobs/{job_id}")
async def get_job_status(job_id: str):
    """Get status of an evaluation job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    return active_jobs[job_id]


@app.get("/api/v1/leaderboard")
async def get_leaderboard(
    benchmark: Optional[str] = None,
    metric: str = "average_score",
    limit: int = 10
):
    """Get model leaderboard."""
    try:
        leaderboard = eval_suite.get_leaderboard(benchmark, metric)
        return {
            "leaderboard": leaderboard[:limit],
            "benchmark": benchmark,
            "metric": metric,
            "total_entries": len(leaderboard)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats")
async def get_stats():
    """Get system statistics."""
    results_history = eval_suite.get_results_history()
    
    # Calculate stats
    total_evaluations = len(results_history)
    total_questions = sum(len(r.benchmark_results) for r in results_history)
    
    # Get unique models and benchmarks
    models = set()
    benchmarks = set()
    for results in results_history:
        for benchmark_result in results.benchmark_results:
            models.add(benchmark_result.model_name)
            benchmarks.add(benchmark_result.benchmark_name)
    
    return {
        "total_evaluations": total_evaluations,
        "total_questions": total_questions,
        "unique_models": len(models),
        "unique_benchmarks": len(benchmarks),
        "active_jobs": len(active_jobs),
        "available_benchmarks": eval_suite.list_benchmarks()
    }


async def run_evaluation(job_id: str, request: dict):
    """Run evaluation in background."""
    try:
        # Update job status
        active_jobs[job_id]["status"] = "running"
        active_jobs[job_id]["started_at"] = datetime.now().isoformat()
        
        # Extract request parameters
        model_name = request["model"]
        provider = request["provider"]
        api_key = request.get("api_key")
        benchmarks = request.get("benchmarks", "all")
        num_questions = request.get("num_questions")
        temperature = request.get("temperature", 0.0)
        max_tokens = request.get("max_tokens", 2048)
        parallel = request.get("parallel", True)
        
        # Create model instance
        model = Model(
            provider=provider,
            name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens
        )
        
        # Parse benchmarks
        if isinstance(benchmarks, str):
            if benchmarks.lower() == "all":
                benchmark_list = "all"
            else:
                benchmark_list = [b.strip() for b in benchmarks.split(",")]
        else:
            benchmark_list = benchmarks
        
        # Run evaluation
        results = await eval_suite.evaluate(
            model=model,
            benchmarks=benchmark_list,
            num_questions=num_questions,
            parallel=parallel,
            save_results=True
        )
        
        # Update job with results
        active_jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.now().isoformat(),
            "result": results.summary()
        })
        
    except Exception as e:
        # Update job with error
        active_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.now().isoformat(),
            "error": str(e)
        })
        logger.error(f"Evaluation job {job_id} failed: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("simple_main:app", host="0.0.0.0", port=8000, reload=True)