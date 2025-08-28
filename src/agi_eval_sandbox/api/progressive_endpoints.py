"""
Progressive Quality Gates API Endpoints
RESTful endpoints for managing progressive quality assessment.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import uuid

from ..core.models import EvaluationContext
from ..quality.progressive_quality_gates import (
    ProgressiveQualityGates,
    ProgressiveConfig,
    DevelopmentPhase,
    RiskLevel,
    ProgressiveQualityResult,
    progressive_quality_gates
)
from ..quality.adaptive_quality_intelligence import (
    AdaptiveQualityIntelligence,
    adaptive_quality_intelligence,
    QualityTrend,
    QualityAnomalyDetection
)
from ..quality.quantum_quality_optimization import (
    QuantumQualityOptimizer,
    quantum_quality_optimizer,
    OptimizationResult,
    OptimizationStrategy
)
from .models import (
    EvaluationRequest,
    EvaluationResponse,
    JobStatus,
    JobResponse
)

router = APIRouter(prefix="/progressive", tags=["progressive-quality"])

# Active progressive evaluation jobs
progressive_jobs: Dict[str, Dict[str, Any]] = {}


class ProgressiveEvaluationRequest(EvaluationRequest):
    """Extended evaluation request with progressive quality configuration."""
    development_phase: str = "development"
    risk_level: str = "medium"
    enable_ml_prediction: bool = True
    enable_adaptive_thresholds: bool = True
    enable_context_awareness: bool = True


class ProgressiveEvaluationResponse(EvaluationResponse):
    """Enhanced evaluation response with progressive insights."""
    development_phase: str
    risk_level: str 
    confidence_score: float
    recommendations: List[str]
    risk_factors: List[str]
    adaptive_thresholds: Dict[str, float]
    ml_predictions: Dict[str, Any]
    phase_advancement: Dict[str, Any]


@router.post("/evaluate", response_model=ProgressiveEvaluationResponse)
async def evaluate_progressive_quality(
    request: ProgressiveEvaluationRequest,
    background_tasks: BackgroundTasks
) -> ProgressiveEvaluationResponse:
    """
    Run progressive quality evaluation with adaptive intelligence.
    
    This endpoint provides context-aware quality assessment that adapts
    based on development phase and risk profile.
    """
    try:
        # Validate phase and risk level
        try:
            phase = DevelopmentPhase(request.development_phase)
            risk = RiskLevel(request.risk_level)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid phase or risk level: {str(e)}"
            )
        
        # Configure progressive gates
        config = ProgressiveConfig(
            phase=phase,
            risk_level=risk,
            enable_ml_quality_prediction=request.enable_ml_prediction,
            enable_adaptive_thresholds=request.enable_adaptive_thresholds,
            enable_context_aware_gates=request.enable_context_awareness
        )
        
        # Create quality gates instance
        gates = ProgressiveQualityGates(config)
        
        # Create evaluation context
        context = EvaluationContext(
            model_name=request.model.name,
            model_provider=request.model.provider,
            benchmarks=request.benchmarks,
            timestamp=datetime.now(),
            metadata={
                "phase": request.development_phase,
                "risk_level": request.risk_level,
                "request_id": str(uuid.uuid4())
            }
        )
        
        # Run progressive evaluation
        result = await gates.evaluate(context)
        
        # Get phase advancement recommendations
        phase_advancement = await gates.get_phase_recommendations()
        
        # Create response
        response = ProgressiveEvaluationResponse(
            id=str(uuid.uuid4()),
            status="completed",
            model=request.model,
            benchmarks=request.benchmarks,
            started_at=datetime.now(),
            completed_at=datetime.now(),
            results={
                "overall_score": result.base_result.overall_score,
                "passed": result.base_result.passed,
                "metrics": [
                    {
                        "name": m.name,
                        "score": m.score,
                        "passed": m.passed,
                        "message": m.message,
                        "duration": m.duration_seconds
                    }
                    for m in result.base_result.metrics
                ]
            },
            development_phase=result.phase.value,
            risk_level=result.risk_level.value,
            confidence_score=result.confidence_score,
            recommendations=result.recommendations,
            risk_factors=result.risk_factors,
            adaptive_thresholds=result.adaptive_thresholds,
            ml_predictions=result.ml_predictions,
            phase_advancement=phase_advancement
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Progressive evaluation failed: {str(e)}"
        )


@router.post("/evaluate/async", response_model=JobResponse)
async def evaluate_progressive_quality_async(
    request: ProgressiveEvaluationRequest,
    background_tasks: BackgroundTasks
) -> JobResponse:
    """
    Start asynchronous progressive quality evaluation.
    
    Returns immediately with job ID for tracking progress.
    """
    job_id = str(uuid.uuid4())
    
    # Store job info
    progressive_jobs[job_id] = {
        "id": job_id,
        "status": "running",
        "started_at": datetime.now(),
        "request": request.dict(),
        "result": None
    }
    
    # Start background evaluation
    background_tasks.add_task(
        _run_progressive_evaluation_background,
        job_id,
        request
    )
    
    return JobResponse(
        id=job_id,
        status="running",
        started_at=datetime.now()
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
async def get_progressive_job_status(job_id: str) -> JobResponse:
    """Get status of progressive evaluation job."""
    if job_id not in progressive_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = progressive_jobs[job_id]
    
    return JobResponse(
        id=job_id,
        status=job["status"],
        started_at=job["started_at"],
        completed_at=job.get("completed_at"),
        result=job.get("result")
    )


@router.get("/jobs/{job_id}/result", response_model=ProgressiveEvaluationResponse)
async def get_progressive_job_result(job_id: str) -> ProgressiveEvaluationResponse:
    """Get result of completed progressive evaluation job."""
    if job_id not in progressive_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = progressive_jobs[job_id]
    
    if job["status"] != "completed":
        raise HTTPException(
            status_code=400, 
            detail=f"Job not completed yet, status: {job['status']}"
        )
    
    if not job.get("result"):
        raise HTTPException(status_code=404, detail="Job result not found")
    
    return job["result"]


@router.put("/phase/{phase}")
async def set_development_phase(phase: str) -> Dict[str, str]:
    """Set global development phase for quality gates."""
    try:
        development_phase = DevelopmentPhase(phase)
        progressive_quality_gates.set_phase(development_phase)
        
        return {
            "message": f"Development phase set to: {phase}",
            "phase": phase
        }
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid phase: {phase}. Valid phases: {[p.value for p in DevelopmentPhase]}"
        )


@router.put("/risk/{risk_level}")
async def set_risk_level(risk_level: str) -> Dict[str, str]:
    """Set global risk level for quality gates."""
    try:
        risk = RiskLevel(risk_level)
        progressive_quality_gates.set_risk_level(risk)
        
        return {
            "message": f"Risk level set to: {risk_level}",
            "risk_level": risk_level
        }
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid risk level: {risk_level}. Valid levels: {[r.value for r in RiskLevel]}"
        )


@router.get("/phases")
async def get_available_phases() -> Dict[str, List[str]]:
    """Get available development phases and risk levels."""
    return {
        "development_phases": [phase.value for phase in DevelopmentPhase],
        "risk_levels": [level.value for level in RiskLevel]
    }


@router.get("/phase/recommendations")
async def get_phase_recommendations() -> Dict[str, Any]:
    """Get recommendations for phase advancement."""
    return await progressive_quality_gates.get_phase_recommendations()


@router.get("/thresholds/{phase}/{risk_level}")
async def get_quality_thresholds(phase: str, risk_level: str) -> Dict[str, float]:
    """Get quality thresholds for specific phase and risk level."""
    try:
        development_phase = DevelopmentPhase(phase)
        risk = RiskLevel(risk_level)
        
        # Create temporary gates instance for threshold lookup
        config = ProgressiveConfig(phase=development_phase, risk_level=risk)
        gates = ProgressiveQualityGates(config)
        
        # Get strategy and thresholds
        strategy = gates.strategies[development_phase]
        thresholds = strategy.get_thresholds(development_phase, risk)
        
        return thresholds
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid phase or risk level: {str(e)}"
        )


@router.get("/performance/history")
async def get_performance_history() -> Dict[str, List[float]]:
    """Get historical performance data for trend analysis."""
    return progressive_quality_gates.performance_history


@router.post("/performance/reset")
async def reset_performance_history() -> Dict[str, str]:
    """Reset performance history (for testing/development)."""
    progressive_quality_gates.performance_history.clear()
    return {"message": "Performance history reset"}


# Adaptive Quality Intelligence Endpoints

@router.get("/intelligence/trends/{metric_name}")
async def get_metric_trend(metric_name: str) -> Dict[str, Any]:
    """Get trend analysis for a specific metric."""
    # Create dummy evaluation context for trend analysis
    context = EvaluationContext(
        model_name="analysis",
        model_provider="system", 
        benchmarks=[],
        timestamp=datetime.now()
    )
    
    # Mock a progressive quality result for trend analysis
    if metric_name in adaptive_quality_intelligence.metric_history:
        history = list(adaptive_quality_intelligence.metric_history[metric_name])
        if history:
            from ..quality.progressive_quality_gates import QualityMetric, QualityGateResult, ProgressiveQualityResult
            
            # Create mock result for analysis
            mock_metrics = [QualityMetric(
                name=metric_name,
                passed=entry["value"] >= 0.7,
                score=entry["value"],
                message="Historical data point"
            ) for entry in history[-5:]]  # Last 5 entries
            
            mock_base_result = QualityGateResult(
                timestamp=datetime.now(),
                passed=True,
                overall_score=sum(m.score for m in mock_metrics) / len(mock_metrics),
                metrics=mock_metrics
            )
            
            mock_progressive_result = ProgressiveQualityResult(
                base_result=mock_base_result,
                phase=DevelopmentPhase.DEVELOPMENT,
                risk_level=RiskLevel.MEDIUM,
                confidence_score=0.8
            )
            
            trends = await adaptive_quality_intelligence.analyze_quality_trends(
                [mock_progressive_result], context
            )
            
            trend = next((t for t in trends if t.metric_name == metric_name), None)
            if trend:
                return {
                    "metric_name": trend.metric_name,
                    "pattern": trend.pattern.value,
                    "trend_strength": trend.trend_strength,
                    "confidence": trend.confidence.value,
                    "predicted_next_values": trend.predicted_next_values,
                    "recommendation": trend.recommendation,
                    "risk_score": trend.risk_score
                }
    
    raise HTTPException(status_code=404, detail=f"No trend data found for metric: {metric_name}")


@router.post("/intelligence/anomalies/detect")
async def detect_anomalies(
    metrics: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Detect anomalies in provided quality metrics."""
    try:
        from ..quality.progressive_quality_gates import QualityMetric
        
        # Convert input to QualityMetric objects
        quality_metrics = []
        for metric_data in metrics:
            quality_metric = QualityMetric(
                name=metric_data.get("name", "unknown"),
                passed=metric_data.get("passed", False),
                score=metric_data.get("score", 0.0),
                message=metric_data.get("message", "")
            )
            quality_metrics.append(quality_metric)
        
        # Create evaluation context
        context = EvaluationContext(
            model_name="anomaly_detection",
            model_provider="system",
            benchmarks=[],
            timestamp=datetime.now()
        )
        
        # Detect anomalies
        anomalies = await adaptive_quality_intelligence.detect_quality_anomalies(
            quality_metrics, context
        )
        
        # Convert to response format
        anomaly_responses = []
        for anomaly in anomalies:
            anomaly_responses.append({
                "metric_name": anomaly.metric_name,
                "current_value": anomaly.current_value,
                "expected_value": anomaly.expected_value,
                "anomaly_score": anomaly.anomaly_score,
                "is_anomaly": anomaly.is_anomaly,
                "severity": anomaly.severity,
                "explanation": anomaly.explanation
            })
        
        return anomaly_responses
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Anomaly detection failed: {str(e)}"
        )


@router.get("/intelligence/summary")
async def get_intelligence_summary() -> Dict[str, Any]:
    """Get summary of adaptive intelligence insights."""
    return await adaptive_quality_intelligence.get_intelligence_summary()


@router.post("/intelligence/state/save")
async def save_intelligence_state() -> Dict[str, str]:
    """Save current intelligence state."""
    try:
        from pathlib import Path
        
        state_dir = Path("/tmp/agi_eval_intelligence")
        state_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        state_file = state_dir / f"intelligence_state_{timestamp}.json"
        
        adaptive_quality_intelligence.save_intelligence_state(state_file)
        
        return {
            "message": "Intelligence state saved successfully",
            "file_path": str(state_file)
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save intelligence state: {str(e)}"
        )


@router.post("/intelligence/state/load")
async def load_intelligence_state(file_path: str) -> Dict[str, str]:
    """Load intelligence state from file."""
    try:
        from pathlib import Path
        
        state_file = Path(file_path)
        if not state_file.exists():
            raise HTTPException(status_code=404, detail="State file not found")
        
        adaptive_quality_intelligence.load_intelligence_state(state_file)
        
        return {
            "message": "Intelligence state loaded successfully",
            "file_path": file_path
        }
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load intelligence state: {str(e)}"
        )


@router.get("/intelligence/metrics")
async def get_tracked_metrics() -> Dict[str, Any]:
    """Get list of metrics being tracked by intelligence system."""
    metrics_info = {}
    
    for metric_name, history in adaptive_quality_intelligence.metric_history.items():
        if history:
            values = [entry["value"] for entry in history]
            metrics_info[metric_name] = {
                "data_points": len(history),
                "latest_value": values[-1] if values else None,
                "average_value": sum(values) / len(values) if values else None,
                "min_value": min(values) if values else None,
                "max_value": max(values) if values else None
            }
    
    return {
        "tracked_metrics": metrics_info,
        "total_metrics": len(metrics_info)
    }


# Quantum Quality Optimization Endpoints

@router.post("/quantum/optimize")
async def optimize_quality_system(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Perform quantum-inspired optimization of quality system."""
    try:
        from ..quality.progressive_quality_gates import QualityMetric, QualityGateResult, ProgressiveQualityResult
        
        # Convert input to ProgressiveQualityResult objects
        progressive_results = []
        for result_data in results:
            # Create mock metrics
            metrics_data = result_data.get("metrics", [])
            quality_metrics = []
            
            for metric_data in metrics_data:
                quality_metric = QualityMetric(
                    name=metric_data.get("name", "unknown"),
                    passed=metric_data.get("passed", False),
                    score=metric_data.get("score", 0.0),
                    message=metric_data.get("message", ""),
                    duration_seconds=metric_data.get("duration_seconds", 0.0)
                )
                quality_metrics.append(quality_metric)
            
            # Create base result
            base_result = QualityGateResult(
                timestamp=datetime.now(),
                passed=result_data.get("passed", False),
                overall_score=result_data.get("overall_score", 0.0),
                metrics=quality_metrics
            )
            
            # Create progressive result
            progressive_result = ProgressiveQualityResult(
                base_result=base_result,
                phase=DevelopmentPhase(result_data.get("phase", "development")),
                risk_level=RiskLevel(result_data.get("risk_level", "medium")),
                confidence_score=result_data.get("confidence_score", 0.7)
            )
            
            progressive_results.append(progressive_result)
        
        # Create evaluation context
        context = EvaluationContext(
            model_name="optimization",
            model_provider="quantum",
            benchmarks=[],
            timestamp=datetime.now()
        )
        
        # Perform quantum optimization
        optimization_result = await quantum_quality_optimizer.optimize_quality_system(
            progressive_results,
            adaptive_quality_intelligence,
            context
        )
        
        # Convert to response format
        return {
            "optimized_thresholds": optimization_result.optimized_thresholds,
            "resource_allocation": optimization_result.resource_allocation,
            "quality_improvement": optimization_result.quality_improvement,
            "energy_function_value": optimization_result.energy_function_value,
            "convergence_iterations": optimization_result.convergence_iterations,
            "quantum_advantage": optimization_result.quantum_advantage,
            "optimization_strategy": optimization_result.optimized_strategy.value,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Quantum optimization failed: {str(e)}"
        )


@router.get("/quantum/status")
async def get_quantum_system_status() -> Dict[str, Any]:
    """Get status of quantum optimization system."""
    return await quantum_quality_optimizer.get_quantum_system_status()


@router.post("/quantum/entanglement/detect")
async def detect_quantum_entanglement(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Detect quantum entanglement between quality metrics."""
    try:
        from ..quality.progressive_quality_gates import QualityMetric, QualityGateResult, ProgressiveQualityResult
        
        # Convert input to ProgressiveQualityResult objects (similar to optimize endpoint)
        progressive_results = []
        for result_data in results:
            metrics_data = result_data.get("metrics", [])
            quality_metrics = []
            
            for metric_data in metrics_data:
                quality_metric = QualityMetric(
                    name=metric_data.get("name", "unknown"),
                    passed=metric_data.get("passed", False),
                    score=metric_data.get("score", 0.0),
                    message=metric_data.get("message", ""),
                    duration_seconds=metric_data.get("duration_seconds", 0.0)
                )
                quality_metrics.append(quality_metric)
            
            base_result = QualityGateResult(
                timestamp=datetime.now(),
                passed=result_data.get("passed", False),
                overall_score=result_data.get("overall_score", 0.0),
                metrics=quality_metrics
            )
            
            progressive_result = ProgressiveQualityResult(
                base_result=base_result,
                phase=DevelopmentPhase(result_data.get("phase", "development")),
                risk_level=RiskLevel(result_data.get("risk_level", "medium")),
                confidence_score=result_data.get("confidence_score", 0.7)
            )
            
            progressive_results.append(progressive_result)
        
        # Detect entanglement
        await quantum_quality_optimizer._detect_quantum_entanglement(progressive_results)
        
        # Return entanglement information
        entanglement_info = {}
        for metric_name, entangled_set in quantum_quality_optimizer.entanglement_graph.items():
            if entangled_set:
                entanglement_info[metric_name] = {
                    "entangled_with": list(entangled_set),
                    "entanglement_count": len(entangled_set)
                }
        
        return {
            "entanglement_graph": entanglement_info,
            "total_entangled_metrics": len(entanglement_info),
            "total_entanglement_pairs": sum(len(entangled) for entangled in quantum_quality_optimizer.entanglement_graph.values()) // 2
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Entanglement detection failed: {str(e)}"
        )


@router.get("/quantum/states")
async def get_quantum_states() -> Dict[str, Any]:
    """Get current quantum states of quality metrics."""
    states_info = {}
    
    for metric_name, state in quantum_quality_optimizer.quantum_states.items():
        states_info[metric_name] = {
            "amplitudes": [{"real": amp.real, "imag": amp.imag} for amp in state.amplitudes],
            "probabilities": state.probabilities,
            "entangled_metrics": state.entangled_metrics,
            "coherence_time": state.coherence_time,
            "fidelity": state.fidelity,
            "superposition_degree": 1.0 - max(state.probabilities) if state.probabilities else 0.0
        }
    
    return {
        "quantum_states": states_info,
        "total_states": len(states_info)
    }


@router.get("/quantum/strategies")
async def get_optimization_strategies() -> Dict[str, List[str]]:
    """Get available quantum optimization strategies."""
    return {
        "optimization_strategies": [strategy.value for strategy in OptimizationStrategy],
        "description": {
            "quantum_annealing": "Quantum annealing for global optimization",
            "variational_optimization": "Variational quantum eigensolver approach",
            "adiabatic_evolution": "Adiabatic quantum evolution",
            "quantum_approximate": "Quantum approximate optimization algorithm", 
            "hybrid_classical": "Hybrid quantum-classical optimization"
        }
    }


async def _run_progressive_evaluation_background(
    job_id: str,
    request: ProgressiveEvaluationRequest
) -> None:
    """Background task for progressive evaluation."""
    try:
        # Configure progressive gates
        phase = DevelopmentPhase(request.development_phase)
        risk = RiskLevel(request.risk_level)
        
        config = ProgressiveConfig(
            phase=phase,
            risk_level=risk,
            enable_ml_quality_prediction=request.enable_ml_prediction,
            enable_adaptive_thresholds=request.enable_adaptive_thresholds,
            enable_context_aware_gates=request.enable_context_awareness
        )
        
        gates = ProgressiveQualityGates(config)
        
        # Create evaluation context
        context = EvaluationContext(
            model_name=request.model.name,
            model_provider=request.model.provider,
            benchmarks=request.benchmarks,
            timestamp=datetime.now(),
            metadata={
                "phase": request.development_phase,
                "risk_level": request.risk_level,
                "job_id": job_id
            }
        )
        
        # Run evaluation
        result = await gates.evaluate(context)
        
        # Get phase advancement recommendations
        phase_advancement = await gates.get_phase_recommendations()
        
        # Create response
        response = ProgressiveEvaluationResponse(
            id=job_id,
            status="completed",
            model=request.model,
            benchmarks=request.benchmarks,
            started_at=progressive_jobs[job_id]["started_at"],
            completed_at=datetime.now(),
            results={
                "overall_score": result.base_result.overall_score,
                "passed": result.base_result.passed,
                "metrics": [
                    {
                        "name": m.name,
                        "score": m.score,
                        "passed": m.passed,
                        "message": m.message,
                        "duration": m.duration_seconds
                    }
                    for m in result.base_result.metrics
                ]
            },
            development_phase=result.phase.value,
            risk_level=result.risk_level.value,
            confidence_score=result.confidence_score,
            recommendations=result.recommendations,
            risk_factors=result.risk_factors,
            adaptive_thresholds=result.adaptive_thresholds,
            ml_predictions=result.ml_predictions,
            phase_advancement=phase_advancement
        )
        
        # Update job
        progressive_jobs[job_id].update({
            "status": "completed",
            "completed_at": datetime.now(),
            "result": response
        })
        
    except Exception as e:
        # Handle errors
        progressive_jobs[job_id].update({
            "status": "failed",
            "completed_at": datetime.now(),
            "error": str(e)
        })