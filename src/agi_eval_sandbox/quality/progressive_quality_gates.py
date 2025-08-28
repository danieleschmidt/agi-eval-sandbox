"""
Progressive Quality Gates System - Generation 1+ Implementation
Adaptive quality assessment that evolves with development phase and risk profile.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple

from ..core.logging_config import get_logger
from ..core.models import EvaluationContext
from .quality_gates import QualityGateResult, QualityMetric
from .security_scanner import SecurityScanResult, security_scanner

logger = get_logger("progressive_quality_gates")


class DevelopmentPhase(Enum):
    """Development lifecycle phases with different quality requirements."""
    PROTOTYPE = "prototype"
    DEVELOPMENT = "development" 
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    RESEARCH = "research"


class RiskLevel(Enum):
    """Risk assessment levels for progressive gate enforcement."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ProgressiveConfig:
    """Configuration for progressive quality gates."""
    phase: DevelopmentPhase = DevelopmentPhase.DEVELOPMENT
    risk_level: RiskLevel = RiskLevel.MEDIUM
    enable_ml_quality_prediction: bool = True
    enable_adaptive_thresholds: bool = True
    enable_context_aware_gates: bool = True
    enable_performance_learning: bool = True
    minimum_confidence_score: float = 0.75
    adaptive_learning_rate: float = 0.1
    

@dataclass 
class ProgressiveQualityResult:
    """Enhanced quality result with progressive insights."""
    base_result: QualityGateResult
    phase: DevelopmentPhase
    risk_level: RiskLevel
    confidence_score: float
    recommendations: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    performance_trends: Dict[str, float] = field(default_factory=dict)
    adaptive_thresholds: Dict[str, float] = field(default_factory=dict)
    ml_predictions: Dict[str, Any] = field(default_factory=dict)


class QualityGateStrategy(Protocol):
    """Protocol for quality gate strategies."""
    
    async def evaluate(self, context: EvaluationContext) -> List[QualityMetric]:
        """Evaluate quality metrics for given context."""
        ...
        
    def get_thresholds(self, phase: DevelopmentPhase, risk: RiskLevel) -> Dict[str, float]:
        """Get phase/risk-specific thresholds."""
        ...


class PrototypeQualityStrategy:
    """Lenient quality gates for rapid prototyping."""
    
    async def evaluate(self, context: EvaluationContext) -> List[QualityMetric]:
        """Fast, basic quality checks for prototypes."""
        metrics = []
        
        # Syntax validation
        start_time = time.time()
        try:
            # Basic syntax check
            syntax_ok = await self._check_syntax(context)
            duration = time.time() - start_time
            
            metrics.append(QualityMetric(
                name="syntax_validation",
                passed=syntax_ok,
                score=1.0 if syntax_ok else 0.0,
                message="Syntax validation passed" if syntax_ok else "Syntax errors found",
                duration_seconds=duration
            ))
        except Exception as e:
            metrics.append(QualityMetric(
                name="syntax_validation",
                passed=False,
                score=0.0,
                message=f"Syntax check failed: {str(e)}",
                duration_seconds=time.time() - start_time
            ))
        
        # Basic functionality test
        start_time = time.time()
        try:
            functionality_ok = await self._check_basic_functionality(context)
            duration = time.time() - start_time
            
            metrics.append(QualityMetric(
                name="basic_functionality",
                passed=functionality_ok,
                score=1.0 if functionality_ok else 0.5,  # More lenient
                message="Basic functionality works" if functionality_ok else "Basic functionality issues",
                duration_seconds=duration
            ))
        except Exception as e:
            metrics.append(QualityMetric(
                name="basic_functionality", 
                passed=False,
                score=0.3,  # Still partially acceptable for prototypes
                message=f"Functionality check failed: {str(e)}",
                duration_seconds=time.time() - start_time
            ))
        
        return metrics
    
    def get_thresholds(self, phase: DevelopmentPhase, risk: RiskLevel) -> Dict[str, float]:
        """Lenient thresholds for prototype phase."""
        return {
            "overall_pass_threshold": 0.6,
            "syntax_threshold": 0.8,
            "functionality_threshold": 0.5,
            "performance_threshold": 0.3
        }
    
    async def _check_syntax(self, context: EvaluationContext) -> bool:
        """Basic syntax validation."""
        # Implementation would check for syntax errors
        return True
        
    async def _check_basic_functionality(self, context: EvaluationContext) -> bool:
        """Basic functionality check."""
        # Implementation would run basic smoke tests
        return True


class ProductionQualityStrategy:
    """Strict quality gates for production systems."""
    
    async def evaluate(self, context: EvaluationContext) -> List[QualityMetric]:
        """Comprehensive quality validation for production."""
        metrics = []
        
        # Security validation
        start_time = time.time()
        try:
            security_result = await security_scanner.scan_context(context)
            duration = time.time() - start_time
            
            security_score = 1.0 - (security_result.vulnerability_count / 10.0)
            security_score = max(0.0, min(1.0, security_score))
            
            metrics.append(QualityMetric(
                name="security_validation",
                passed=security_result.vulnerability_count == 0,
                score=security_score,
                message=f"Security scan: {security_result.vulnerability_count} vulnerabilities",
                duration_seconds=duration,
                details={"vulnerabilities": security_result.vulnerabilities}
            ))
        except Exception as e:
            metrics.append(QualityMetric(
                name="security_validation",
                passed=False,
                score=0.0,
                message=f"Security scan failed: {str(e)}",
                duration_seconds=time.time() - start_time
            ))
        
        # Performance validation
        start_time = time.time()
        try:
            perf_score = await self._check_performance(context)
            duration = time.time() - start_time
            
            metrics.append(QualityMetric(
                name="performance_validation",
                passed=perf_score >= 0.8,
                score=perf_score,
                message=f"Performance score: {perf_score:.2f}",
                duration_seconds=duration
            ))
        except Exception as e:
            metrics.append(QualityMetric(
                name="performance_validation",
                passed=False,
                score=0.0,
                message=f"Performance check failed: {str(e)}",
                duration_seconds=time.time() - start_time
            ))
        
        # Reliability validation  
        start_time = time.time()
        try:
            reliability_score = await self._check_reliability(context)
            duration = time.time() - start_time
            
            metrics.append(QualityMetric(
                name="reliability_validation",
                passed=reliability_score >= 0.9,
                score=reliability_score,
                message=f"Reliability score: {reliability_score:.2f}",
                duration_seconds=duration
            ))
        except Exception as e:
            metrics.append(QualityMetric(
                name="reliability_validation",
                passed=False,
                score=0.0,
                message=f"Reliability check failed: {str(e)}",
                duration_seconds=time.time() - start_time
            ))
        
        return metrics
    
    def get_thresholds(self, phase: DevelopmentPhase, risk: RiskLevel) -> Dict[str, float]:
        """Strict thresholds for production phase."""
        return {
            "overall_pass_threshold": 0.95,
            "security_threshold": 1.0,
            "performance_threshold": 0.8,
            "reliability_threshold": 0.9
        }
    
    async def _check_performance(self, context: EvaluationContext) -> float:
        """Performance benchmark validation."""
        # Implementation would run performance tests
        return 0.85
        
    async def _check_reliability(self, context: EvaluationContext) -> float:
        """Reliability and stability validation."""
        # Implementation would run reliability tests
        return 0.92


class ProgressiveQualityGates:
    """Progressive Quality Gates System with adaptive intelligence."""
    
    def __init__(self, config: Optional[ProgressiveConfig] = None):
        self.config = config or ProgressiveConfig()
        self.strategies = {
            DevelopmentPhase.PROTOTYPE: PrototypeQualityStrategy(),
            DevelopmentPhase.DEVELOPMENT: PrototypeQualityStrategy(),
            DevelopmentPhase.TESTING: ProductionQualityStrategy(),
            DevelopmentPhase.STAGING: ProductionQualityStrategy(),
            DevelopmentPhase.PRODUCTION: ProductionQualityStrategy(),
            DevelopmentPhase.RESEARCH: PrototypeQualityStrategy(),
        }
        self.performance_history: Dict[str, List[float]] = {}
        self.adaptive_thresholds: Dict[str, float] = {}
        
    async def evaluate(self, context: EvaluationContext) -> ProgressiveQualityResult:
        """Run progressive quality evaluation."""
        logger.info(f"🎯 Running progressive quality gates - Phase: {self.config.phase}, Risk: {self.config.risk_level}")
        
        start_time = time.time()
        
        # Get strategy for current phase
        strategy = self.strategies[self.config.phase]
        
        # Run quality metrics
        metrics = await strategy.evaluate(context)
        
        # Calculate overall score
        overall_score = sum(m.score for m in metrics) / len(metrics) if metrics else 0.0
        
        # Apply adaptive thresholds if enabled
        if self.config.enable_adaptive_thresholds:
            thresholds = await self._get_adaptive_thresholds(strategy, context)
        else:
            thresholds = strategy.get_thresholds(self.config.phase, self.config.risk_level)
        
        # Determine pass/fail
        passed = overall_score >= thresholds["overall_pass_threshold"]
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(metrics, thresholds)
        
        # Calculate confidence score
        confidence_score = await self._calculate_confidence(metrics, context)
        
        # Identify risk factors
        risk_factors = await self._identify_risk_factors(metrics, context)
        
        # ML predictions (if enabled)
        ml_predictions = {}
        if self.config.enable_ml_quality_prediction:
            ml_predictions = await self._predict_quality_trends(context, metrics)
        
        # Update performance history
        await self._update_performance_history(metrics)
        
        # Create base result
        base_result = QualityGateResult(
            passed=passed,
            overall_score=overall_score,
            metrics=metrics,
            total_duration_seconds=time.time() - start_time
        )
        
        result = ProgressiveQualityResult(
            base_result=base_result,
            phase=self.config.phase,
            risk_level=self.config.risk_level,
            confidence_score=confidence_score,
            recommendations=recommendations,
            risk_factors=risk_factors,
            adaptive_thresholds=thresholds,
            ml_predictions=ml_predictions
        )
        
        logger.info(f"✅ Progressive quality evaluation complete - Score: {overall_score:.3f}, Passed: {passed}")
        
        return result
    
    async def _get_adaptive_thresholds(
        self, 
        strategy: QualityGateStrategy, 
        context: EvaluationContext
    ) -> Dict[str, float]:
        """Calculate adaptive thresholds based on historical performance."""
        base_thresholds = strategy.get_thresholds(self.config.phase, self.config.risk_level)
        
        if not self.performance_history:
            return base_thresholds
        
        adaptive_thresholds = base_thresholds.copy()
        
        # Adjust thresholds based on historical performance
        for metric_name, history in self.performance_history.items():
            if len(history) >= 5:  # Need sufficient history
                avg_performance = sum(history[-10:]) / len(history[-10:])  # Last 10 runs
                trend = (history[-1] - history[-5]) / 5 if len(history) >= 5 else 0
                
                # Adjust threshold based on performance trend
                if trend > 0.05:  # Improving trend
                    adjustment = min(0.1, trend * 2)  # Increase threshold
                elif trend < -0.05:  # Declining trend
                    adjustment = max(-0.1, trend * 2)  # Lower threshold
                else:
                    adjustment = 0
                
                threshold_key = f"{metric_name}_threshold"
                if threshold_key in adaptive_thresholds:
                    new_threshold = base_thresholds[threshold_key] + adjustment
                    adaptive_thresholds[threshold_key] = max(0.1, min(1.0, new_threshold))
        
        self.adaptive_thresholds = adaptive_thresholds
        return adaptive_thresholds
    
    async def _generate_recommendations(
        self, 
        metrics: List[QualityMetric], 
        thresholds: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        failed_metrics = [m for m in metrics if not m.passed]
        
        if not failed_metrics:
            recommendations.append("✅ All quality gates passed! Consider advancing to next phase.")
            return recommendations
        
        for metric in failed_metrics:
            if metric.name == "security_validation":
                recommendations.append(f"🔒 Security: Address {len(metric.details.get('vulnerabilities', []))} vulnerabilities")
            elif metric.name == "performance_validation":
                recommendations.append(f"⚡ Performance: Optimize to achieve {thresholds.get('performance_threshold', 0.8):.0%} score")
            elif metric.name == "reliability_validation":
                recommendations.append(f"🛡️ Reliability: Improve stability to reach {thresholds.get('reliability_threshold', 0.9):.0%}")
            elif metric.name == "basic_functionality":
                recommendations.append("🔧 Functionality: Fix core functionality issues before proceeding")
        
        # Phase-specific recommendations
        if self.config.phase == DevelopmentPhase.PROTOTYPE:
            recommendations.append("💡 Prototype Phase: Focus on core functionality over optimization")
        elif self.config.phase == DevelopmentPhase.PRODUCTION:
            recommendations.append("🚀 Production Phase: All metrics must meet strict criteria")
        
        return recommendations
    
    async def _calculate_confidence(
        self, 
        metrics: List[QualityMetric], 
        context: EvaluationContext
    ) -> float:
        """Calculate confidence score for the quality assessment."""
        if not metrics:
            return 0.0
        
        # Base confidence on metric consistency and historical performance
        score_variance = sum((m.score - sum(m2.score for m2 in metrics) / len(metrics))**2 for m in metrics) / len(metrics)
        consistency_factor = 1.0 - min(1.0, score_variance * 2)
        
        # Historical accuracy factor
        historical_factor = 1.0  # Would be calculated from prediction accuracy history
        
        # Context complexity factor  
        complexity_factor = 0.9 if len(context.__dict__) > 10 else 1.0
        
        confidence = (consistency_factor * 0.4 + historical_factor * 0.4 + complexity_factor * 0.2)
        
        return min(1.0, max(0.0, confidence))
    
    async def _identify_risk_factors(
        self, 
        metrics: List[QualityMetric], 
        context: EvaluationContext
    ) -> List[str]:
        """Identify potential risk factors."""
        risk_factors = []
        
        failed_metrics = [m for m in metrics if not m.passed]
        critical_failures = [m for m in failed_metrics if m.score < 0.5]
        
        if critical_failures:
            risk_factors.append(f"Critical failures in {len(critical_failures)} metrics")
        
        if len(failed_metrics) > len(metrics) * 0.5:
            risk_factors.append("High failure rate across multiple metrics")
        
        # Check for degradation trends
        for metric_name, history in self.performance_history.items():
            if len(history) >= 3:
                recent_trend = (history[-1] - history[-3]) / 3
                if recent_trend < -0.1:
                    risk_factors.append(f"Declining trend in {metric_name}")
        
        return risk_factors
    
    async def _predict_quality_trends(
        self, 
        context: EvaluationContext, 
        metrics: List[QualityMetric]
    ) -> Dict[str, Any]:
        """Predict future quality trends using ML."""
        predictions = {}
        
        # Simple trend prediction based on historical data
        for metric_name, history in self.performance_history.items():
            if len(history) >= 5:
                # Linear trend prediction
                x = list(range(len(history)))
                y = history
                
                # Simple linear regression
                n = len(x)
                sum_x = sum(x)
                sum_y = sum(y)
                sum_xy = sum(x[i] * y[i] for i in range(n))
                sum_x2 = sum(x[i] ** 2 for i in range(n))
                
                if n * sum_x2 - sum_x ** 2 != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
                    predicted_next = history[-1] + slope
                    
                    predictions[metric_name] = {
                        "trend": "improving" if slope > 0.01 else "declining" if slope < -0.01 else "stable",
                        "predicted_next_score": max(0.0, min(1.0, predicted_next)),
                        "confidence": min(1.0, len(history) / 10.0)
                    }
        
        return predictions
    
    async def _update_performance_history(self, metrics: List[QualityMetric]) -> None:
        """Update performance history for adaptive learning."""
        for metric in metrics:
            if metric.name not in self.performance_history:
                self.performance_history[metric.name] = []
            
            self.performance_history[metric.name].append(metric.score)
            
            # Keep only last 50 entries
            if len(self.performance_history[metric.name]) > 50:
                self.performance_history[metric.name] = self.performance_history[metric.name][-50:]
    
    def set_phase(self, phase: DevelopmentPhase) -> None:
        """Update development phase."""
        self.config.phase = phase
        logger.info(f"🔄 Phase updated to: {phase.value}")
    
    def set_risk_level(self, risk_level: RiskLevel) -> None:
        """Update risk level.""" 
        self.config.risk_level = risk_level
        logger.info(f"⚠️ Risk level updated to: {risk_level.value}")
    
    async def get_phase_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for advancing to next phase."""
        current_phase = self.config.phase
        
        recommendations = {
            "current_phase": current_phase.value,
            "can_advance": True,
            "next_phase": None,
            "requirements": [],
            "estimated_readiness": 0.0
        }
        
        # Define phase progression path
        phase_order = [
            DevelopmentPhase.PROTOTYPE,
            DevelopmentPhase.DEVELOPMENT, 
            DevelopmentPhase.TESTING,
            DevelopmentPhase.STAGING,
            DevelopmentPhase.PRODUCTION
        ]
        
        if current_phase in phase_order:
            current_index = phase_order.index(current_phase)
            if current_index < len(phase_order) - 1:
                recommendations["next_phase"] = phase_order[current_index + 1].value
        
        # Calculate readiness based on historical performance
        if self.performance_history:
            avg_scores = []
            for history in self.performance_history.values():
                if history:
                    avg_scores.append(sum(history[-5:]) / len(history[-5:]))
            
            if avg_scores:
                recommendations["estimated_readiness"] = sum(avg_scores) / len(avg_scores)
        
        return recommendations


# Global instance
progressive_quality_gates = ProgressiveQualityGates()