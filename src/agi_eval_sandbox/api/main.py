"""Main FastAPI application."""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Any
import asyncio
import logging
import uuid
from datetime import datetime

from ..core import EvalSuite, Model
from ..core.context_compressor import ContextCompressionEngine, CompressionStrategy, CompressionConfig
from ..config import settings
from .models import (
    EvaluationRequest,
    EvaluationResponse,
    ModelSpec,
    BenchmarkInfo,
    LeaderboardEntry,
    JobStatus,
    JobResponse,
    ComparisonRequest,
    StatsResponse,
    CustomBenchmarkRequest,
    CustomBenchmarkResponse
)
from .progressive_endpoints import router as progressive_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
eval_suite = EvalSuite()
compression_engine = ContextCompressionEngine()
active_jobs: Dict[str, Dict[str, Any]] = {}
custom_benchmarks: Dict[str, Dict[str, Any]] = {}
compression_jobs: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("🚀 Starting AGI Evaluation Sandbox API")
    logger.info("🔧 Initializing context compression engine...")
    await compression_engine.initialize()
    logger.info("✅ Context compression engine ready")
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

# Include progressive quality gates router
app.include_router(progressive_router, prefix="/api/v1")


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


@app.get("/api/v1")
async def api_v1_root():
    """API v1 root endpoint with information."""
    return {
        "message": "AGI Evaluation Sandbox API v1",
        "version": "0.1.0",
        "docs": "/docs",
        "endpoints": {
            "evaluate": "/api/v1/evaluate",
            "jobs": "/api/v1/jobs",
            "leaderboard": "/api/v1/leaderboard",
            "benchmarks": "/api/v1/benchmarks",
            "custom_benchmarks": "/api/v1/benchmarks/custom",
            "compression": "/api/v1/compress",
            "stats": "/api/v1/stats"
        }
    }


# API v1 endpoints
@app.get("/api/v1/benchmarks", response_model=List[BenchmarkInfo])
async def list_benchmarks():
    """List available benchmarks."""
    benchmarks = []
    for name in eval_suite.list_benchmarks():
        benchmark = eval_suite.get_benchmark(name)
        if benchmark:
            questions = benchmark.get_questions()
            
            # Get categories
            categories = list(set(q.category for q in questions if q.category))
            
            benchmarks.append(BenchmarkInfo(
                name=name,
                version=benchmark.version,
                total_questions=len(questions),
                categories=categories,
                description=f"{name.upper()} benchmark with {len(questions)} questions"
            ))
    
    return benchmarks


@app.get("/api/v1/benchmarks/{benchmark_name}")
async def get_benchmark_details(benchmark_name: str):
    """Get detailed information about a specific benchmark."""
    benchmark = eval_suite.get_benchmark(benchmark_name)
    if not benchmark:
        raise HTTPException(status_code=404, detail=f"Benchmark '{benchmark_name}' not found")
    
    questions = benchmark.get_questions()
    
    # Analyze question types and categories
    question_types = {}
    categories = {}
    difficulties = {}
    
    for question in questions:
        # Count question types
        q_type = question.question_type.value
        question_types[q_type] = question_types.get(q_type, 0) + 1
        
        # Count categories
        if question.category:
            categories[question.category] = categories.get(question.category, 0) + 1
        
        # Count difficulties
        if question.difficulty:
            difficulties[question.difficulty] = difficulties.get(question.difficulty, 0) + 1
    
    # Sample questions
    sample_questions = questions[:5]
    samples = []
    for q in sample_questions:
        sample = {
            "id": q.id,
            "prompt": q.prompt[:200] + "..." if len(q.prompt) > 200 else q.prompt,
            "type": q.question_type.value,
            "category": q.category,
            "difficulty": q.difficulty
        }
        if q.choices:
            sample["choices"] = q.choices
        samples.append(sample)
    
    return {
        "name": benchmark_name,
        "version": benchmark.version,
        "total_questions": len(questions),
        "question_types": question_types,
        "categories": categories,
        "difficulties": difficulties,
        "sample_questions": samples
    }


@app.post("/api/v1/benchmarks/custom", response_model=CustomBenchmarkResponse)
async def create_custom_benchmark(request: CustomBenchmarkRequest):
    """Create custom benchmark."""
    try:
        # Generate unique benchmark ID
        benchmark_id = f"custom_{str(uuid.uuid4())}"
        
        # Validate questions format
        for i, question in enumerate(request.questions):
            if "prompt" not in question:
                raise HTTPException(
                    status_code=400,
                    detail=f"Question {i+1} missing required 'prompt' field"
                )
        
        # Store custom benchmark
        custom_benchmarks[benchmark_id] = {
            "id": benchmark_id,
            "name": request.name,
            "description": request.description,
            "questions": request.questions,
            "category": request.category or "custom",
            "tags": request.tags,
            "created_at": datetime.now(),
            "total_questions": len(request.questions)
        }
        
        logger.info(f"Created custom benchmark '{request.name}' with ID {benchmark_id}")
        
        return CustomBenchmarkResponse(
            benchmark_id=benchmark_id,
            name=request.name,
            status="created",
            message="Custom benchmark created successfully",
            total_questions=len(request.questions)
        )
        
    except Exception as e:
        logger.error(f"Failed to create custom benchmark: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/v1/benchmarks/custom")
async def list_custom_benchmarks():
    """List all custom benchmarks."""
    benchmarks = []
    for benchmark_id, benchmark_data in custom_benchmarks.items():
        benchmarks.append({
            "benchmark_id": benchmark_id,
            "name": benchmark_data["name"],
            "description": benchmark_data["description"],
            "total_questions": benchmark_data["total_questions"],
            "category": benchmark_data["category"],
            "tags": benchmark_data["tags"],
            "created_at": benchmark_data["created_at"].isoformat()
        })
    
    # Sort by creation time (newest first)
    benchmarks.sort(key=lambda x: x["created_at"], reverse=True)
    return {"custom_benchmarks": benchmarks}


@app.get("/api/v1/benchmarks/custom/{benchmark_id}")
async def get_custom_benchmark(benchmark_id: str):
    """Get details of a specific custom benchmark."""
    if benchmark_id not in custom_benchmarks:
        raise HTTPException(status_code=404, detail="Custom benchmark not found")
    
    return custom_benchmarks[benchmark_id]


@app.delete("/api/v1/benchmarks/custom/{benchmark_id}")
async def delete_custom_benchmark(benchmark_id: str):
    """Delete a custom benchmark."""
    if benchmark_id not in custom_benchmarks:
        raise HTTPException(status_code=404, detail="Custom benchmark not found")
    
    benchmark_name = custom_benchmarks[benchmark_id]["name"]
    del custom_benchmarks[benchmark_id]
    
    return {
        "message": f"Custom benchmark '{benchmark_name}' deleted successfully",
        "benchmark_id": benchmark_id
    }


@app.post("/api/v1/evaluate", response_model=JobResponse)
async def start_evaluation(
    request: EvaluationRequest,
    background_tasks: BackgroundTasks
):
    """Submit evaluation job."""
    try:
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Create model instance
        model = Model(
            provider=request.model.provider,
            name=request.model.name,
            api_key=request.model.api_key,
            temperature=request.config.temperature,
            max_tokens=request.config.max_tokens
        )
        
        # Initialize job tracking
        active_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.now(),
            "model": request.model.dict(),
            "benchmarks": request.benchmarks,
            "config": request.config.dict(),
            "results": None,
            "error": None
        }
        
        # Start evaluation in background
        background_tasks.add_task(
            run_evaluation_job,
            job_id,
            model,
            request.benchmarks,
            request.config.dict()
        )
        
        return JobResponse(
            job_id=job_id,
            status="pending",
            message="Evaluation job started"
        )
        
    except Exception as e:
        logger.error(f"Failed to start evaluation: {e}")
        raise HTTPException(status_code=400, detail=str(e))


async def run_evaluation_job(
    job_id: str,
    model: Model,
    benchmarks: List[str],
    config: Dict[str, Any]
):
    """Run evaluation job in background."""
    try:
        active_jobs[job_id]["status"] = "running"
        active_jobs[job_id]["progress"] = 0.1
        
        # Run evaluation
        benchmark_list = benchmarks if benchmarks != ["all"] else "all"
        results = await eval_suite.evaluate(
            model=model,
            benchmarks=benchmark_list,
            save_results=False,
            **config
        )
        
        # Update job with results
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["progress"] = 1.0
        active_jobs[job_id]["results"] = results.summary()
        active_jobs[job_id]["completed_at"] = datetime.now()
        
        logger.info(f"Evaluation job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Evaluation job {job_id} failed: {e}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)


@app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get job status."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    return JobStatus(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        created_at=job["created_at"],
        completed_at=job.get("completed_at"),
        error=job.get("error")
    )


@app.get("/api/v1/jobs/{job_id}/results")
async def get_job_results(job_id: str):
    """Get results of a completed evaluation job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed (status: {job['status']})"
        )
    
    return {
        "job_id": job_id,
        "results": job["results"],
        "model": job["model"],
        "benchmarks": job["benchmarks"],
        "config": job["config"]
    }


@app.get("/api/v1/jobs")
async def list_jobs():
    """List all evaluation jobs."""
    jobs = []
    for job_id, job_data in active_jobs.items():
        jobs.append({
            "job_id": job_id,
            "status": job_data["status"],
            "created_at": job_data["created_at"].isoformat(),
            "model": job_data["model"]["name"],
            "provider": job_data["model"]["provider"],
            "benchmarks": job_data["benchmarks"]
        })
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"jobs": jobs}


@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(job_id: str):
    """Cancel or delete an evaluation job."""
    if job_id not in active_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = active_jobs[job_id]
    
    if job["status"] == "running":
        # In a real implementation, you would cancel the background task
        job["status"] = "cancelled"
        return {"message": "Job cancelled"}
    else:
        # Delete completed/failed jobs
        del active_jobs[job_id]
        return {"message": "Job deleted"}


@app.get("/api/v1/leaderboard", response_model=List[LeaderboardEntry])
async def get_leaderboard(
    benchmark: Optional[str] = None,
    metric: str = "average_score",
    limit: int = 50
):
    """Get model leaderboard."""
    try:
        leaderboard_data = eval_suite.get_leaderboard(
            benchmark=benchmark,
            metric=metric
        )
        
        entries = []
        for i, record in enumerate(leaderboard_data[:limit]):
            entries.append(LeaderboardEntry(
                rank=i + 1,
                model_name=record["model_name"],
                model_provider=record["model_provider"],
                benchmark=record["benchmark"],
                average_score=record["average_score"],
                pass_rate=record["pass_rate"],
                total_questions=record["total_questions"],
                timestamp=record["timestamp"]
            ))
        
        return entries
        
    except Exception as e:
        logger.error(f"Failed to get leaderboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve leaderboard")


@app.post("/api/v1/compare")
async def compare_models(
    request: ComparisonRequest,
    background_tasks: BackgroundTasks
):
    """Compare multiple models on the same benchmarks."""
    try:
        models_config = [model.dict() for model in request.models]
        benchmarks = request.benchmarks
        config = request.config.dict()
        
        if len(models_config) < 2:
            raise HTTPException(
                status_code=400,
                detail="At least 2 models required for comparison"
            )
        
        # Create job ID for comparison
        job_id = str(uuid.uuid4())
        
        # Create model instances
        models = []
        for model_config in models_config:
            model = Model(
                provider=model_config["provider"],
                name=model_config["name"],
                api_key=model_config.get("api_key")
            )
            models.append(model)
        
        # Initialize comparison job
        active_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.now(),
            "type": "comparison",
            "models": models_config,
            "benchmarks": benchmarks,
            "config": config,
            "results": None,
            "error": None
        }
        
        # Start comparison in background
        background_tasks.add_task(
            run_comparison_job,
            job_id,
            models,
            benchmarks,
            config
        )
        
        return JobResponse(
            job_id=job_id,
            status="pending",
            message="Model comparison started"
        )
        
    except Exception as e:
        logger.error(f"Failed to start comparison: {e}")
        raise HTTPException(status_code=400, detail=str(e))


async def run_comparison_job(
    job_id: str,
    models: List[Model],
    benchmarks: List[str],
    config: Dict[str, Any]
):
    """Run model comparison in background."""
    try:
        active_jobs[job_id]["status"] = "running"
        active_jobs[job_id]["progress"] = 0.1
        
        # Run comparison
        benchmark_list = benchmarks if benchmarks != ["all"] else "all"
        comparison_results = eval_suite.compare_models(
            models=models,
            benchmarks=benchmark_list,
            **config
        )
        
        # Format results
        formatted_results = {}
        for model_name, results in comparison_results.items():
            formatted_results[model_name] = results.summary()
        
        # Update job with results
        active_jobs[job_id]["status"] = "completed"
        active_jobs[job_id]["progress"] = 1.0
        active_jobs[job_id]["results"] = formatted_results
        active_jobs[job_id]["completed_at"] = datetime.now()
        
        logger.info(f"Comparison job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Comparison job {job_id} failed: {e}")
        active_jobs[job_id]["status"] = "failed"
        active_jobs[job_id]["error"] = str(e)


# Context Compression Endpoints

@app.get("/api/v1/compress/strategies")
async def list_compression_strategies():
    """List all available compression strategies."""
    strategies = []
    for strategy in CompressionStrategy:
        descriptions = {
            CompressionStrategy.EXTRACTIVE_SUMMARIZATION: "Select most important sentences based on importance scoring",
            CompressionStrategy.SENTENCE_CLUSTERING: "Group similar sentences and select representatives from each cluster", 
            CompressionStrategy.SEMANTIC_FILTERING: "Filter sentences by semantic similarity to key topics",
            CompressionStrategy.TOKEN_PRUNING: "Remove less important tokens while preserving meaning",
            CompressionStrategy.IMPORTANCE_SAMPLING: "Probabilistic sampling based on comprehensive importance scores",
            CompressionStrategy.HIERARCHICAL_COMPRESSION: "Multi-level compression with document structure analysis"
        }
        
        strategies.append({
            "name": strategy.value,
            "description": descriptions.get(strategy, "Advanced compression technique"),
            "available": strategy in compression_engine.get_available_strategies()
        })
    
    return {"strategies": strategies}


@app.post("/api/v1/compress")
async def compress_text(
    background_tasks: BackgroundTasks,
    text: str,
    strategy: Optional[str] = "extractive_summarization",
    target_ratio: Optional[float] = 0.5,
    target_length: Optional[int] = None,
    preserve_structure: Optional[bool] = True,
    model_name: Optional[str] = "sentence-transformers/all-MiniLM-L6-v2",
    semantic_threshold: Optional[float] = 0.7
):
    """Compress text using specified strategy."""
    try:
        # Validate inputs
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if target_ratio and not (0.1 <= target_ratio <= 0.9):
            raise HTTPException(status_code=400, detail="Target ratio must be between 0.1 and 0.9")
        
        try:
            compression_strategy = CompressionStrategy(strategy)
        except ValueError:
            available_strategies = [s.value for s in CompressionStrategy]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid strategy '{strategy}'. Available: {available_strategies}"
            )
        
        # Create job ID
        job_id = str(uuid.uuid4())
        
        # Initialize job tracking
        compression_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.now(),
            "strategy": strategy,
            "config": {
                "target_ratio": target_ratio,
                "target_length": target_length,
                "preserve_structure": preserve_structure,
                "model_name": model_name,
                "semantic_threshold": semantic_threshold
            },
            "original_length": len(text),
            "results": None,
            "error": None
        }
        
        # Start compression in background
        background_tasks.add_task(
            run_compression_job,
            job_id,
            text,
            compression_strategy,
            target_ratio,
            target_length,
            preserve_structure,
            model_name,
            semantic_threshold
        )
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Compression job started",
            "strategy": strategy
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start compression: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


async def run_compression_job(
    job_id: str,
    text: str,
    strategy: CompressionStrategy,
    target_ratio: Optional[float],
    target_length: Optional[int],
    preserve_structure: bool,
    model_name: str,
    semantic_threshold: float
):
    """Run compression job in background."""
    try:
        compression_jobs[job_id]["status"] = "running"
        compression_jobs[job_id]["progress"] = 0.1
        
        # Configure compression
        config = CompressionConfig(
            strategy=strategy,
            target_ratio=target_ratio or 0.5,
            semantic_threshold=semantic_threshold,
            model_name=model_name
        )
        
        # Update engine config
        compression_engine.config = config
        
        # Run compression
        compressed_text, metrics = await compression_engine.compress(
            text=text,
            strategy=strategy,
            target_length=target_length,
            preserve_structure=preserve_structure
        )
        
        # Update job with results
        compression_jobs[job_id]["status"] = "completed"
        compression_jobs[job_id]["progress"] = 1.0
        compression_jobs[job_id]["results"] = {
            "compressed_text": compressed_text,
            "metrics": {
                "original_tokens": metrics.original_tokens,
                "compressed_tokens": metrics.compressed_tokens,
                "compression_ratio": metrics.compression_ratio,
                "processing_time": metrics.processing_time,
                "semantic_similarity": metrics.semantic_similarity,
                "information_retention": metrics.information_retention
            }
        }
        compression_jobs[job_id]["completed_at"] = datetime.now()
        
        logger.info(f"Compression job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Compression job {job_id} failed: {e}")
        compression_jobs[job_id]["status"] = "failed"
        compression_jobs[job_id]["error"] = str(e)


@app.get("/api/v1/compress/jobs/{job_id}")
async def get_compression_job_status(job_id: str):
    """Get compression job status."""
    if job_id not in compression_jobs:
        raise HTTPException(status_code=404, detail="Compression job not found")
    
    job = compression_jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "created_at": job["created_at"].isoformat(),
        "strategy": job["strategy"],
        "config": job["config"],
        "original_length": job["original_length"]
    }
    
    if job.get("completed_at"):
        response["completed_at"] = job["completed_at"].isoformat()
    
    if job.get("error"):
        response["error"] = job["error"]
    
    if job.get("results"):
        response["results"] = job["results"]
    
    return response


@app.get("/api/v1/compress/jobs/{job_id}/results")
async def get_compression_results(job_id: str):
    """Get results of a completed compression job."""
    if job_id not in compression_jobs:
        raise HTTPException(status_code=404, detail="Compression job not found")
    
    job = compression_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job is not completed (status: {job['status']})"
        )
    
    return {
        "job_id": job_id,
        "compressed_text": job["results"]["compressed_text"],
        "metrics": job["results"]["metrics"],
        "strategy": job["strategy"],
        "config": job["config"]
    }


@app.get("/api/v1/compress/jobs")
async def list_compression_jobs():
    """List all compression jobs."""
    jobs = []
    for job_id, job_data in compression_jobs.items():
        jobs.append({
            "job_id": job_id,
            "status": job_data["status"],
            "created_at": job_data["created_at"].isoformat(),
            "strategy": job_data["strategy"],
            "original_length": job_data["original_length"]
        })
    
    # Sort by creation time (newest first)
    jobs.sort(key=lambda x: x["created_at"], reverse=True)
    return {"compression_jobs": jobs}


@app.delete("/api/v1/compress/jobs/{job_id}")
async def delete_compression_job(job_id: str):
    """Delete a compression job."""
    if job_id not in compression_jobs:
        raise HTTPException(status_code=404, detail="Compression job not found")
    
    job = compression_jobs[job_id]
    
    if job["status"] == "running":
        # Cancel running job
        job["status"] = "cancelled"
        return {"message": "Compression job cancelled"}
    else:
        # Delete completed/failed jobs
        del compression_jobs[job_id]
        return {"message": "Compression job deleted"}


@app.post("/api/v1/compress/benchmark")
async def benchmark_compression_strategies(
    background_tasks: BackgroundTasks,
    text: str,
    target_length: Optional[int] = None
):
    """Benchmark all available compression strategies on the same text."""
    try:
        if not text or not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        # Create job ID for benchmark
        job_id = str(uuid.uuid4())
        
        # Initialize benchmark job
        compression_jobs[job_id] = {
            "status": "pending",
            "progress": 0.0,
            "created_at": datetime.now(),
            "type": "benchmark",
            "original_length": len(text),
            "target_length": target_length,
            "results": None,
            "error": None
        }
        
        # Start benchmark in background
        background_tasks.add_task(
            run_compression_benchmark_job,
            job_id,
            text,
            target_length
        )
        
        return {
            "job_id": job_id,
            "status": "pending",
            "message": "Compression benchmark started",
            "type": "benchmark"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start compression benchmark: {e}")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


async def run_compression_benchmark_job(
    job_id: str,
    text: str,
    target_length: Optional[int]
):
    """Run compression benchmark job in background."""
    try:
        compression_jobs[job_id]["status"] = "running"
        compression_jobs[job_id]["progress"] = 0.1
        
        # Run benchmark
        benchmark_results = await compression_engine.benchmark_strategies(text, target_length)
        
        # Format results
        formatted_results = {}
        for strategy, (compressed_text, metrics) in benchmark_results.items():
            formatted_results[strategy.value] = {
                "compressed_text": compressed_text,
                "metrics": {
                    "original_tokens": metrics.original_tokens,
                    "compressed_tokens": metrics.compressed_tokens,
                    "compression_ratio": metrics.compression_ratio,
                    "processing_time": metrics.processing_time,
                    "semantic_similarity": metrics.semantic_similarity,
                    "information_retention": metrics.information_retention
                }
            }
        
        # Update job with results
        compression_jobs[job_id]["status"] = "completed"
        compression_jobs[job_id]["progress"] = 1.0
        compression_jobs[job_id]["results"] = formatted_results
        compression_jobs[job_id]["completed_at"] = datetime.now()
        
        logger.info(f"Compression benchmark job {job_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Compression benchmark job {job_id} failed: {e}")
        compression_jobs[job_id]["status"] = "failed"
        compression_jobs[job_id]["error"] = str(e)


@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats():
    """Get API usage statistics."""
    total_jobs = len(active_jobs)
    completed_jobs = sum(1 for job in active_jobs.values() if job["status"] == "completed")
    running_jobs = sum(1 for job in active_jobs.values() if job["status"] == "running")
    failed_jobs = sum(1 for job in active_jobs.values() if job["status"] == "failed")
    
    total_compression_jobs = len(compression_jobs)
    completed_compression_jobs = sum(1 for job in compression_jobs.values() if job["status"] == "completed")
    
    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "running_jobs": running_jobs,
        "failed_jobs": failed_jobs,
        "total_compression_jobs": total_compression_jobs,
        "completed_compression_jobs": completed_compression_jobs,
        "available_benchmarks": len(eval_suite.list_benchmarks()) + len(custom_benchmarks),
        "available_compression_strategies": len(compression_engine.get_available_strategies()),
        "uptime": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)