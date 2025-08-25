"""Auto-scaling system for dynamic resource management."""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)


@dataclass
class ScalingMetrics:
    """Metrics for scaling decisions."""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_requests: int = 0
    queue_size: int = 0
    response_time_p95: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0


class ScalingAction(Enum):
    """Scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingRule:
    """Scaling rule definition."""
    name: str
    metric: str
    threshold: float
    action: ScalingAction
    cooldown_seconds: int = 300


class AutoScaler:
    """Auto-scaling system for dynamic resource management."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scaling_rules = self._create_default_rules()
        self._last_scale_time = 0
        
    def _create_default_rules(self) -> List[ScalingRule]:
        """Create default scaling rules."""
        return [
            ScalingRule("cpu_high", "cpu_usage", 80.0, ScalingAction.SCALE_UP),
            ScalingRule("memory_high", "memory_usage", 85.0, ScalingAction.SCALE_UP),
            ScalingRule("queue_high", "queue_size", 50, ScalingAction.SCALE_UP),
            ScalingRule("cpu_low", "cpu_usage", 20.0, ScalingAction.SCALE_DOWN),
            ScalingRule("memory_low", "memory_usage", 30.0, ScalingAction.SCALE_DOWN),
        ]
    
    def should_scale_up(self, metrics: ScalingMetrics) -> bool:
        """Check if system should scale up."""
        scale_up_rules = [r for r in self.scaling_rules if r.action == ScalingAction.SCALE_UP]
        
        for rule in scale_up_rules:
            metric_value = getattr(metrics, rule.metric, 0)
            if metric_value > rule.threshold:
                logger.info(f"Scale up triggered by rule: {rule.name}")
                return True
        
        return False
    
    def should_scale_down(self, metrics: ScalingMetrics) -> bool:
        """Check if system should scale down."""
        scale_down_rules = [r for r in self.scaling_rules if r.action == ScalingAction.SCALE_DOWN]
        
        # Only scale down if ALL conditions are met
        for rule in scale_down_rules:
            metric_value = getattr(metrics, rule.metric, 100)  # Default high for safety
            if metric_value > rule.threshold:
                return False
        
        logger.info("Scale down conditions met")
        return True
    
    def get_scaling_recommendations(self, metrics: ScalingMetrics) -> List[str]:
        """Get scaling recommendations based on current metrics."""
        recommendations = []
        
        if self.should_scale_up(metrics):
            recommendations.append("Scale up: High resource utilization detected")
            recommendations.append(f"CPU: {metrics.cpu_usage}%, Memory: {metrics.memory_usage}%")
            
        if self.should_scale_down(metrics):
            recommendations.append("Scale down: Low resource utilization")
            recommendations.append("Consider reducing instance count to save costs")
            
        if metrics.error_rate > 5.0:
            recommendations.append("High error rate detected - investigate before scaling")
            
        return recommendations
