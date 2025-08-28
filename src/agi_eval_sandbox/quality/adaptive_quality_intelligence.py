"""
Adaptive Quality Intelligence System - Generation 2+ Implementation
AI-driven quality assessment with machine learning and predictive analytics.
"""

import asyncio
import json
import numpy as np
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from collections import defaultdict, deque

from ..core.logging_config import get_logger
from ..core.models import EvaluationContext
from .progressive_quality_gates import (
    ProgressiveQualityResult,
    DevelopmentPhase,
    RiskLevel,
    QualityMetric
)

logger = get_logger("adaptive_quality_intelligence")


class QualityPattern(Enum):
    """Patterns detected in quality metrics."""
    IMPROVING = "improving"
    STABLE = "stable"
    DECLINING = "declining"
    OSCILLATING = "oscillating"
    ANOMALOUS = "anomalous"


class PredictionConfidence(Enum):
    """Confidence levels for quality predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class QualityTrend:
    """Quality trend analysis result."""
    metric_name: str
    pattern: QualityPattern
    trend_strength: float  # -1.0 to 1.0
    confidence: PredictionConfidence
    predicted_next_values: List[float]
    recommendation: str
    risk_score: float  # 0.0 to 1.0


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive quality intelligence."""
    enable_ml_prediction: bool = True
    enable_anomaly_detection: bool = True
    enable_pattern_recognition: bool = True
    enable_risk_modeling: bool = True
    history_window_size: int = 50
    trend_analysis_window: int = 10
    anomaly_threshold: float = 2.0  # Standard deviations
    prediction_horizon: int = 5
    learning_rate: float = 0.01
    confidence_threshold: float = 0.7


@dataclass
class QualityAnomalyDetection:
    """Anomaly detection result."""
    metric_name: str
    current_value: float
    expected_value: float
    anomaly_score: float
    is_anomaly: bool
    severity: str  # "low", "medium", "high", "critical"
    explanation: str


class AdaptiveQualityIntelligence:
    """AI-driven adaptive quality intelligence system."""
    
    def __init__(self, config: Optional[AdaptiveConfig] = None):
        self.config = config or AdaptiveConfig()
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.config.history_window_size))
        self.pattern_models: Dict[str, Dict[str, Any]] = {}
        self.anomaly_models: Dict[str, Dict[str, Any]] = {}
        self.risk_models: Dict[str, Dict[str, Any]] = {}
        self.prediction_accuracy: Dict[str, deque] = defaultdict(lambda: deque(maxlen=20))
        self.last_predictions: Dict[str, List[float]] = {}
        
        # Statistical tracking
        self.baseline_metrics: Dict[str, Dict[str, float]] = {}
        self.seasonal_patterns: Dict[str, List[float]] = {}
        
        # Initialize models
        asyncio.create_task(self._initialize_models())
    
    async def _initialize_models(self) -> None:
        """Initialize ML models and statistical baselines."""
        logger.info("🧠 Initializing adaptive quality intelligence models")
        
        # Initialize baseline statistical models for common metrics
        common_metrics = [
            "security_validation", "performance_validation", "reliability_validation",
            "syntax_validation", "basic_functionality", "overall_score"
        ]
        
        for metric in common_metrics:
            self.pattern_models[metric] = self._create_pattern_model()
            self.anomaly_models[metric] = self._create_anomaly_model()
            self.risk_models[metric] = self._create_risk_model()
        
        logger.info("✅ Adaptive quality intelligence models initialized")
    
    def _create_pattern_model(self) -> Dict[str, Any]:
        """Create pattern recognition model for a metric."""
        return {
            "type": "statistical_pattern",
            "moving_avg_window": 5,
            "trend_window": 10,
            "seasonal_window": 20,
            "volatility_threshold": 0.1,
            "trained": False
        }
    
    def _create_anomaly_model(self) -> Dict[str, Any]:
        """Create anomaly detection model for a metric."""
        return {
            "type": "statistical_anomaly",
            "mean": 0.0,
            "std": 1.0,
            "quartiles": [0.25, 0.5, 0.75],
            "iqr_multiplier": 1.5,
            "z_score_threshold": self.config.anomaly_threshold,
            "trained": False
        }
    
    def _create_risk_model(self) -> Dict[str, Any]:
        """Create risk assessment model for a metric."""
        return {
            "type": "risk_scoring",
            "baseline_risk": 0.3,
            "volatility_factor": 0.4,
            "trend_factor": 0.3,
            "trained": False
        }
    
    async def analyze_quality_trends(
        self, 
        results: List[ProgressiveQualityResult],
        context: EvaluationContext
    ) -> List[QualityTrend]:
        """Analyze quality trends with AI-driven insights."""
        logger.info("📈 Analyzing quality trends with adaptive intelligence")
        
        trends = []
        
        # Update metric history
        for result in results:
            for metric in result.base_result.metrics:
                self.metric_history[metric.name].append({
                    "timestamp": result.base_result.timestamp,
                    "value": metric.score,
                    "passed": metric.passed,
                    "phase": result.phase.value,
                    "risk_level": result.risk_level.value
                })
        
        # Analyze trends for each metric
        for metric_name, history in self.metric_history.items():
            if len(history) >= 3:  # Need minimum data for analysis
                trend = await self._analyze_metric_trend(metric_name, history, context)
                if trend:
                    trends.append(trend)
        
        # Update prediction accuracy
        await self._update_prediction_accuracy(results)
        
        logger.info(f"📊 Generated {len(trends)} quality trend analyses")
        return trends
    
    async def _analyze_metric_trend(
        self, 
        metric_name: str, 
        history: deque,
        context: EvaluationContext
    ) -> Optional[QualityTrend]:
        """Analyze trend for a specific metric."""
        if len(history) < 3:
            return None
        
        values = [entry["value"] for entry in history]
        timestamps = [entry["timestamp"] for entry in history]
        
        # Pattern recognition
        pattern = await self._detect_pattern(values, metric_name)
        
        # Trend strength calculation
        trend_strength = await self._calculate_trend_strength(values)
        
        # Confidence assessment
        confidence = await self._assess_prediction_confidence(metric_name, values)
        
        # Future predictions
        predicted_values = await self._predict_future_values(
            metric_name, 
            values, 
            self.config.prediction_horizon
        )
        
        # Risk assessment
        risk_score = await self._calculate_risk_score(metric_name, values, pattern)
        
        # Generate recommendation
        recommendation = await self._generate_trend_recommendation(
            metric_name, pattern, trend_strength, risk_score, context
        )
        
        return QualityTrend(
            metric_name=metric_name,
            pattern=pattern,
            trend_strength=trend_strength,
            confidence=confidence,
            predicted_next_values=predicted_values,
            recommendation=recommendation,
            risk_score=risk_score
        )
    
    async def _detect_pattern(self, values: List[float], metric_name: str) -> QualityPattern:
        """Detect quality pattern using statistical analysis."""
        if len(values) < 5:
            return QualityPattern.STABLE
        
        # Calculate trend indicators
        recent_values = values[-self.config.trend_analysis_window:]
        trend_slope = await self._calculate_linear_trend(recent_values)
        volatility = np.std(recent_values) if len(recent_values) > 1 else 0.0
        
        # Pattern detection logic
        if abs(trend_slope) < 0.01 and volatility < 0.05:
            return QualityPattern.STABLE
        elif trend_slope > 0.02:
            return QualityPattern.IMPROVING
        elif trend_slope < -0.02:
            return QualityPattern.DECLINING
        elif volatility > 0.15:
            return QualityPattern.OSCILLATING
        
        # Anomaly detection
        if await self._is_anomalous_pattern(values, metric_name):
            return QualityPattern.ANOMALOUS
        
        return QualityPattern.STABLE
    
    async def _calculate_linear_trend(self, values: List[float]) -> float:
        """Calculate linear trend slope."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x = np.arange(n)
        y = np.array(values)
        
        # Simple linear regression
        slope = np.corrcoef(x, y)[0, 1] * (np.std(y) / np.std(x)) if np.std(x) > 0 else 0.0
        return slope
    
    async def _calculate_trend_strength(self, values: List[float]) -> float:
        """Calculate trend strength (-1.0 to 1.0)."""
        if len(values) < 3:
            return 0.0
        
        # Use correlation coefficient as trend strength
        x = np.arange(len(values))
        y = np.array(values)
        
        try:
            correlation = np.corrcoef(x, y)[0, 1]
            return float(correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    async def _assess_prediction_confidence(
        self, 
        metric_name: str, 
        values: List[float]
    ) -> PredictionConfidence:
        """Assess confidence in predictions."""
        # Base confidence on historical prediction accuracy
        if metric_name in self.prediction_accuracy and len(self.prediction_accuracy[metric_name]) > 5:
            avg_accuracy = sum(self.prediction_accuracy[metric_name]) / len(self.prediction_accuracy[metric_name])
            
            if avg_accuracy >= 0.9:
                return PredictionConfidence.VERY_HIGH
            elif avg_accuracy >= 0.75:
                return PredictionConfidence.HIGH
            elif avg_accuracy >= 0.6:
                return PredictionConfidence.MEDIUM
            else:
                return PredictionConfidence.LOW
        
        # Base confidence on data stability
        if len(values) < 5:
            return PredictionConfidence.LOW
        
        volatility = np.std(values[-10:]) if len(values) >= 10 else np.std(values)
        
        if volatility < 0.05:
            return PredictionConfidence.HIGH
        elif volatility < 0.15:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    async def _predict_future_values(
        self, 
        metric_name: str, 
        values: List[float], 
        horizon: int
    ) -> List[float]:
        """Predict future values using trend analysis."""
        if len(values) < 3:
            return [values[-1]] * horizon if values else [0.5] * horizon
        
        # Simple trend projection
        recent_trend = await self._calculate_linear_trend(values[-min(10, len(values)):])
        last_value = values[-1]
        
        predictions = []
        for i in range(1, horizon + 1):
            # Apply trend with some decay
            decay_factor = 0.9 ** i  # Trend decays over time
            predicted_value = last_value + (recent_trend * i * decay_factor)
            
            # Clamp to reasonable bounds
            predicted_value = max(0.0, min(1.0, predicted_value))
            predictions.append(predicted_value)
        
        # Store predictions for accuracy tracking
        self.last_predictions[metric_name] = predictions
        
        return predictions
    
    async def _calculate_risk_score(
        self, 
        metric_name: str, 
        values: List[float], 
        pattern: QualityPattern
    ) -> float:
        """Calculate risk score for metric."""
        base_risk = 0.3  # Base risk level
        
        # Pattern-based risk adjustments
        pattern_risk = {
            QualityPattern.IMPROVING: -0.2,
            QualityPattern.STABLE: 0.0,
            QualityPattern.DECLINING: 0.3,
            QualityPattern.OSCILLATING: 0.2,
            QualityPattern.ANOMALOUS: 0.4
        }
        
        # Current performance risk
        current_value = values[-1] if values else 0.5
        performance_risk = (1.0 - current_value) * 0.5
        
        # Volatility risk
        volatility = np.std(values[-10:]) if len(values) >= 10 else np.std(values) if len(values) > 1 else 0.0
        volatility_risk = min(volatility * 2.0, 0.3)
        
        total_risk = base_risk + pattern_risk.get(pattern, 0.0) + performance_risk + volatility_risk
        
        return max(0.0, min(1.0, total_risk))
    
    async def _generate_trend_recommendation(
        self,
        metric_name: str,
        pattern: QualityPattern,
        trend_strength: float,
        risk_score: float,
        context: EvaluationContext
    ) -> str:
        """Generate actionable recommendation based on trend analysis."""
        recommendations = []
        
        # Pattern-specific recommendations
        if pattern == QualityPattern.IMPROVING:
            recommendations.append(f"✅ {metric_name} is improving consistently")
            if trend_strength > 0.7:
                recommendations.append("Continue current practices to maintain improvement")
        
        elif pattern == QualityPattern.DECLINING:
            recommendations.append(f"⚠️ {metric_name} is declining - immediate attention needed")
            if trend_strength < -0.5:
                recommendations.append("Critical: Implement corrective actions urgently")
        
        elif pattern == QualityPattern.OSCILLATING:
            recommendations.append(f"📊 {metric_name} shows instability - stabilization needed")
            recommendations.append("Review process consistency and environment factors")
        
        elif pattern == QualityPattern.ANOMALOUS:
            recommendations.append(f"🔍 {metric_name} shows anomalous behavior - investigate root cause")
            recommendations.append("Check for environmental changes or configuration issues")
        
        # Risk-based recommendations
        if risk_score > 0.7:
            recommendations.append(f"🚨 High risk detected for {metric_name}")
            recommendations.append("Consider implementing additional monitoring and controls")
        
        elif risk_score > 0.5:
            recommendations.append(f"⚠️ Medium risk for {metric_name} - monitor closely")
        
        # Context-specific recommendations
        if context.metadata.get("phase") == "production":
            if pattern in [QualityPattern.DECLINING, QualityPattern.ANOMALOUS]:
                recommendations.append("Production phase: Consider rollback or hotfix")
        
        return " | ".join(recommendations) if recommendations else "No specific recommendations at this time"
    
    async def _is_anomalous_pattern(self, values: List[float], metric_name: str) -> bool:
        """Detect anomalous patterns in metric values."""
        if len(values) < 5:
            return False
        
        # Get or create anomaly model
        if metric_name not in self.anomaly_models:
            self.anomaly_models[metric_name] = self._create_anomaly_model()
        
        model = self.anomaly_models[metric_name]
        
        # Update model with recent data
        await self._update_anomaly_model(model, values)
        
        # Check latest value for anomaly
        latest_value = values[-1]
        z_score = abs((latest_value - model["mean"]) / model["std"]) if model["std"] > 0 else 0.0
        
        return z_score > model["z_score_threshold"]
    
    async def _update_anomaly_model(self, model: Dict[str, Any], values: List[float]) -> None:
        """Update anomaly detection model with new data."""
        if len(values) < 2:
            return
        
        # Update statistical parameters
        model["mean"] = np.mean(values)
        model["std"] = np.std(values)
        model["quartiles"] = [np.percentile(values, q) for q in [25, 50, 75]]
        model["trained"] = True
    
    async def detect_quality_anomalies(
        self,
        current_metrics: List[QualityMetric],
        context: EvaluationContext
    ) -> List[QualityAnomalyDetection]:
        """Detect anomalies in current quality metrics."""
        if not self.config.enable_anomaly_detection:
            return []
        
        logger.info("🔍 Detecting quality anomalies with adaptive intelligence")
        
        anomalies = []
        
        for metric in current_metrics:
            if metric.name in self.metric_history and len(self.metric_history[metric.name]) >= 5:
                anomaly = await self._detect_metric_anomaly(metric, context)
                if anomaly and anomaly.is_anomaly:
                    anomalies.append(anomaly)
        
        logger.info(f"🚨 Detected {len(anomalies)} quality anomalies")
        return anomalies
    
    async def _detect_metric_anomaly(
        self,
        metric: QualityMetric,
        context: EvaluationContext
    ) -> Optional[QualityAnomalyDetection]:
        """Detect anomaly for a specific metric."""
        history = self.metric_history[metric.name]
        if len(history) < 5:
            return None
        
        historical_values = [entry["value"] for entry in history]
        
        # Statistical anomaly detection
        mean_val = np.mean(historical_values)
        std_val = np.std(historical_values)
        
        if std_val == 0:
            return None
        
        z_score = abs((metric.score - mean_val) / std_val)
        is_anomaly = z_score > self.config.anomaly_threshold
        
        if not is_anomaly:
            return None
        
        # Determine severity
        if z_score > 4.0:
            severity = "critical"
        elif z_score > 3.0:
            severity = "high"
        elif z_score > 2.5:
            severity = "medium"
        else:
            severity = "low"
        
        # Generate explanation
        explanation = await self._generate_anomaly_explanation(
            metric, mean_val, std_val, z_score, context
        )
        
        return QualityAnomalyDetection(
            metric_name=metric.name,
            current_value=metric.score,
            expected_value=mean_val,
            anomaly_score=z_score,
            is_anomaly=is_anomaly,
            severity=severity,
            explanation=explanation
        )
    
    async def _generate_anomaly_explanation(
        self,
        metric: QualityMetric,
        expected_value: float,
        std_val: float,
        z_score: float,
        context: EvaluationContext
    ) -> str:
        """Generate explanation for detected anomaly."""
        explanations = []
        
        if metric.score > expected_value:
            explanations.append(f"Unusually high score ({metric.score:.3f} vs expected {expected_value:.3f})")
        else:
            explanations.append(f"Unusually low score ({metric.score:.3f} vs expected {expected_value:.3f})")
        
        explanations.append(f"Deviation: {z_score:.1f} standard deviations")
        
        # Context-based explanations
        phase = context.metadata.get("phase", "unknown")
        if phase in ["production", "staging"]:
            explanations.append(f"Critical concern for {phase} environment")
        
        # Time-based explanations
        current_hour = datetime.now().hour
        if 0 <= current_hour <= 6:
            explanations.append("May be related to off-hours processing differences")
        
        return " | ".join(explanations)
    
    async def _update_prediction_accuracy(self, results: List[ProgressiveQualityResult]) -> None:
        """Update prediction accuracy tracking."""
        for result in results:
            for metric in result.base_result.metrics:
                metric_name = metric.name
                
                if metric_name in self.last_predictions:
                    # Compare first prediction with actual value
                    predicted = self.last_predictions[metric_name][0]
                    actual = metric.score
                    
                    # Calculate accuracy (1.0 - normalized error)
                    error = abs(predicted - actual)
                    accuracy = max(0.0, 1.0 - error)
                    
                    self.prediction_accuracy[metric_name].append(accuracy)
    
    async def get_intelligence_summary(self) -> Dict[str, Any]:
        """Get summary of adaptive intelligence insights."""
        summary = {
            "metrics_tracked": len(self.metric_history),
            "total_data_points": sum(len(history) for history in self.metric_history.values()),
            "models_trained": sum(1 for models in self.pattern_models.values() if models.get("trained")),
            "average_prediction_accuracy": {},
            "risk_distribution": {"low": 0, "medium": 0, "high": 0, "critical": 0},
            "pattern_distribution": {pattern.value: 0 for pattern in QualityPattern}
        }
        
        # Calculate average prediction accuracy
        for metric_name, accuracies in self.prediction_accuracy.items():
            if accuracies:
                summary["average_prediction_accuracy"][metric_name] = sum(accuracies) / len(accuracies)
        
        return summary
    
    def save_intelligence_state(self, file_path: Path) -> None:
        """Save intelligence state to file."""
        state = {
            "config": self.config.__dict__,
            "metric_history": {k: list(v) for k, v in self.metric_history.items()},
            "pattern_models": self.pattern_models,
            "anomaly_models": self.anomaly_models,
            "risk_models": self.risk_models,
            "prediction_accuracy": {k: list(v) for k, v in self.prediction_accuracy.items()},
            "last_predictions": self.last_predictions,
            "baseline_metrics": self.baseline_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(file_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        logger.info(f"💾 Intelligence state saved to {file_path}")
    
    def load_intelligence_state(self, file_path: Path) -> None:
        """Load intelligence state from file."""
        try:
            with open(file_path, 'r') as f:
                state = json.load(f)
            
            # Restore state
            self.metric_history = {
                k: deque(v, maxlen=self.config.history_window_size) 
                for k, v in state.get("metric_history", {}).items()
            }
            self.pattern_models = state.get("pattern_models", {})
            self.anomaly_models = state.get("anomaly_models", {})
            self.risk_models = state.get("risk_models", {})
            self.prediction_accuracy = {
                k: deque(v, maxlen=20) 
                for k, v in state.get("prediction_accuracy", {}).items()
            }
            self.last_predictions = state.get("last_predictions", {})
            self.baseline_metrics = state.get("baseline_metrics", {})
            
            logger.info(f"📥 Intelligence state loaded from {file_path}")
            
        except Exception as e:
            logger.warning(f"Failed to load intelligence state: {str(e)}")


# Global instance
adaptive_quality_intelligence = AdaptiveQualityIntelligence()