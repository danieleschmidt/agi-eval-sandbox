"""Pydantic models for API requests and responses."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime
from enum import Enum


class ModelProvider(str, Enum):
    """Supported model providers."""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"


class JobStatusEnum(str, Enum):
    """Job status options."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ModelSpec(BaseModel):
    """Model specification for evaluation."""
    provider: ModelProvider
    name: str
    api_key: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "provider": "openai",
                "name": "gpt-4",
                "api_key": "sk-..."
            }
        }


class EvaluationConfig(BaseModel):
    """Configuration for evaluation runs."""
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)
    num_questions: Optional[int] = Field(default=None, ge=1)
    parallel: bool = Field(default=True)
    seed: Optional[int] = None
    
    class Config:
        schema_extra = {
            "example": {
                "temperature": 0.0,
                "max_tokens": 2048,
                "num_questions": 10,
                "parallel": True,
                "seed": 42
            }
        }


class EvaluationRequest(BaseModel):
    """Request to start an evaluation."""
    model: ModelSpec
    benchmarks: List[str] = Field(default=["all"])
    config: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    class Config:
        schema_extra = {
            "example": {
                "model": {
                    "provider": "openai",
                    "name": "gpt-4",
                    "api_key": "sk-..."
                },
                "benchmarks": ["truthfulqa", "mmlu"],
                "config": {
                    "temperature": 0.0,
                    "max_tokens": 2048,
                    "num_questions": 5
                }
            }
        }


class EvaluationResponse(BaseModel):
    """Response from evaluation."""
    run_id: str
    timestamp: str
    total_benchmarks: int
    total_questions: int
    overall_score: float
    overall_pass_rate: float
    benchmark_scores: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class BenchmarkInfo(BaseModel):
    """Information about a benchmark."""
    name: str
    version: str
    total_questions: int
    categories: List[str]
    description: str
    
    class Config:
        schema_extra = {
            "example": {
                "name": "truthfulqa",
                "version": "1.0",
                "total_questions": 3,
                "categories": ["biology", "geography", "physics"],
                "description": "TruthfulQA benchmark with 3 questions"
            }
        }


class LeaderboardEntry(BaseModel):
    """Entry in the model leaderboard."""
    rank: int
    model_name: str
    model_provider: str
    benchmark: str
    average_score: float
    pass_rate: float
    total_questions: int
    timestamp: str
    
    class Config:
        schema_extra = {
            "example": {
                "rank": 1,
                "model_name": "gpt-4",
                "model_provider": "openai",
                "benchmark": "truthfulqa",
                "average_score": 0.875,
                "pass_rate": 87.5,
                "total_questions": 8,
                "timestamp": "2024-01-15T10:30:00"
            }
        }


class JobStatus(BaseModel):
    """Status of an evaluation job."""
    job_id: str
    status: JobStatusEnum
    progress: float = Field(ge=0.0, le=1.0)
    created_at: datetime
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "running",
                "progress": 0.65,
                "created_at": "2024-01-15T10:30:00Z",
                "completed_at": None,
                "error": None
            }
        }


class JobResponse(BaseModel):
    """Response when starting a job."""
    job_id: str
    status: JobStatusEnum
    message: str
    
    class Config:
        schema_extra = {
            "example": {
                "job_id": "550e8400-e29b-41d4-a716-446655440000",
                "status": "pending",
                "message": "Evaluation job started"
            }
        }


class ComparisonRequest(BaseModel):
    """Request to compare multiple models."""
    models: List[ModelSpec] = Field(min_items=2)
    benchmarks: List[str] = Field(default=["all"])
    config: EvaluationConfig = Field(default_factory=EvaluationConfig)
    
    class Config:
        schema_extra = {
            "example": {
                "models": [
                    {
                        "provider": "openai",
                        "name": "gpt-4",
                        "api_key": "sk-..."
                    },
                    {
                        "provider": "anthropic",
                        "name": "claude-3-opus",
                        "api_key": "sk-ant-..."
                    }
                ],
                "benchmarks": ["truthfulqa", "mmlu"],
                "config": {
                    "temperature": 0.0,
                    "num_questions": 5
                }
            }
        }


class StatsResponse(BaseModel):
    """API usage statistics."""
    total_jobs: int
    completed_jobs: int
    running_jobs: int
    failed_jobs: int
    available_benchmarks: int
    uptime: str
    
    class Config:
        schema_extra = {
            "example": {
                "total_jobs": 42,
                "completed_jobs": 38,
                "running_jobs": 2,
                "failed_jobs": 2,
                "available_benchmarks": 3,
                "uptime": "2024-01-15T10:30:00Z"
            }
        }