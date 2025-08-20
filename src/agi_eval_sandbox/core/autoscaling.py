"""Advanced auto-scaling and resource management for AGI Evaluation Sandbox - Generation 3 Optimized Implementation."""

import asyncio
import time
import threading
import statistics
import math
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict
import json
import logging
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from .logging_config import get_logger
from .exceptions import ResourceError

logger = get_logger("autoscaling")


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    SCALE_OUT = "scale_out"  # Horizontal scaling
    SCALE_IN = "scale_in"    # Horizontal scaling reduction
    OPTIMIZE = "optimize"    # Resource optimization
    NO_ACTION = "no_action"


class ScalingStrategy(Enum):
    """Scaling strategies."""
    REACTIVE = "reactive"      # React to current metrics
    PREDICTIVE = "predictive"  # Predict future load
    HYBRID = "hybrid"          # Combination of reactive and predictive
    ML_BASED = "ml_based"      # Machine learning based scaling


@dataclass
class AdvancedScalingMetrics:
    """Comprehensive metrics for advanced scaling decisions."""
    timestamp: datetime = field(default_factory=datetime.now)
    
    # System metrics
    cpu_utilization: float = 0.0
    cpu_load_1m: float = 0.0
    cpu_load_5m: float = 0.0
    memory_utilization: float = 0.0
    memory_pressure: float = 0.0  # Memory pressure indicator
    disk_io_utilization: float = 0.0
    network_io_utilization: float = 0.0
    
    # Application metrics
    active_requests: int = 0
    queue_length: int = 0
    pending_evaluations: int = 0
    concurrent_evaluations: int = 0
    
    # Performance metrics
    response_time_avg: float = 0.0
    response_time_p50: float = 0.0
    response_time_p95: float = 0.0
    response_time_p99: float = 0.0
    throughput_qps: float = 0.0
    
    # Quality metrics
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    success_rate: float = 100.0
    
    # Business metrics
    cost_per_request: float = 0.0
    user_satisfaction_score: float = 1.0
    
    # Predictive indicators
    load_trend: float = 0.0        # Positive = increasing, negative = decreasing
    seasonal_factor: float = 1.0   # Seasonal load multiplier
    predicted_load_5m: float = 0.0 # Predicted load in 5 minutes
    predicted_load_15m: float = 0.0 # Predicted load in 15 minutes


# Legacy alias for backward compatibility
ScalingMetrics = AdvancedScalingMetrics


@dataclass
class AdvancedScalingRule:
    """Advanced scaling rule with multiple conditions and actions."""
    name: str
    priority: int = 1  # Higher priority rules are evaluated first
    enabled: bool = True
    
    # Conditions (all must be met for rule to trigger)
    conditions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Actions to take when conditions are met
    action: ScalingAction = ScalingAction.NO_ACTION
    scale_factor: float = 1.5
    target_instances: Optional[int] = None  # Explicit target instead of factor
    
    # Constraints
    max_instances: int = 10
    min_instances: int = 1
    cooldown_seconds: int = 300
    
    # Advanced features
    warmup_time: int = 60        # Time to wait for new instances to warm up
    prediction_horizon: int = 300 # How far ahead to predict (seconds)
    confidence_threshold: float = 0.7  # Minimum confidence for prediction-based scaling
    
    # Resource optimization
    cost_optimization: bool = False
    performance_optimization: bool = True
    
    def evaluate_conditions(self, metrics: AdvancedScalingMetrics) -> bool:
        """Evaluate if all conditions are met."""
        if not self.enabled:
            return False
        
        for condition in self.conditions:
            metric_name = condition.get("metric")
            operator = condition.get("operator", ">=")
            threshold = condition.get("threshold")
            
            if not metric_name or threshold is None:
                continue
            
            metric_value = getattr(metrics, metric_name, 0)
            
            if operator == ">=":
                if not (metric_value >= threshold):
                    return False
            elif operator == "<=":
                if not (metric_value <= threshold):
                    return False
            elif operator == ">":
                if not (metric_value > threshold):
                    return False
            elif operator == "<":
                if not (metric_value < threshold):
                    return False
            elif operator == "==":
                if not (abs(metric_value - threshold) < 0.01):
                    return False
        
        return True


# Legacy alias
ScalingRule = AdvancedScalingRule


class PredictiveLoadForecaster:
    """Machine learning-based load forecasting for predictive scaling."""
    
    def __init__(self, history_window: int = 1440):  # 24 hours at 1-minute intervals
        self.logger = logging.getLogger("load_forecaster")
        self.history_window = history_window
        self.metrics_history: deque = deque(maxlen=history_window)
        self.seasonal_patterns: Dict[str, List[float]] = {
            "hourly": [1.0] * 24,    # Hourly patterns
            "daily": [1.0] * 7,     # Daily patterns
            "weekly": [1.0] * 4     # Weekly patterns
        }
        self.trend_weights = deque(maxlen=60)  # Last hour for trend calculation
        
    def add_metrics(self, metrics: AdvancedScalingMetrics):
        """Add metrics to history for learning."""
        self.metrics_history.append(metrics)
        
        # Update trend weights
        if len(self.metrics_history) > 1:
            current_load = self._calculate_load_score(metrics)
            prev_load = self._calculate_load_score(self.metrics_history[-2])
            trend = (current_load - prev_load) / max(prev_load, 0.01)
            self.trend_weights.append(trend)
        
        # Update seasonal patterns periodically
        if len(self.metrics_history) % 60 == 0:  # Every hour
            self._update_seasonal_patterns()
    
    def _calculate_load_score(self, metrics: AdvancedScalingMetrics) -> float:
        """Calculate a composite load score from metrics."""
        weights = {
            "cpu": 0.3,
            "memory": 0.2,
            "queue": 0.3,
            "response_time": 0.2
        }
        
        score = (
            weights["cpu"] * metrics.cpu_utilization +
            weights["memory"] * metrics.memory_utilization +
            weights["queue"] * min(100, metrics.queue_length) +
            weights["response_time"] * min(100, metrics.response_time_p95 / 100)
        )
        
        return score
    
    def _update_seasonal_patterns(self):
        """Update seasonal patterns based on historical data."""
        if len(self.metrics_history) < 168:  # Need at least a week
            return
        
        try:
            # Update hourly patterns
            hourly_loads = [0.0] * 24
            hourly_counts = [0] * 24
            
            for metrics in list(self.metrics_history)[-168:]:  # Last week
                hour = metrics.timestamp.hour
                load = self._calculate_load_score(metrics)
                hourly_loads[hour] += load
                hourly_counts[hour] += 1
            
            # Calculate averages
            for i in range(24):
                if hourly_counts[i] > 0:
                    self.seasonal_patterns["hourly"][i] = hourly_loads[i] / hourly_counts[i]
            
            # Normalize to average of 1.0
            avg_load = sum(self.seasonal_patterns["hourly"]) / 24
            if avg_load > 0:
                self.seasonal_patterns["hourly"] = [
                    load / avg_load for load in self.seasonal_patterns["hourly"]
                ]
            
        except Exception as e:
            self.logger.error(f"Error updating seasonal patterns: {e}")
    
    def predict_load(self, horizon_minutes: int = 5) -> Tuple[float, float]:
        """Predict load for the next horizon_minutes with confidence."""
        if len(self.metrics_history) < 10:
            return 0.0, 0.0  # Not enough data
        
        try:
            # Get current load and trend
            current_metrics = self.metrics_history[-1]
            current_load = self._calculate_load_score(current_metrics)
            
            # Calculate trend
            recent_trend = statistics.mean(self.trend_weights) if self.trend_weights else 0.0
            
            # Apply seasonal factors
            future_time = current_metrics.timestamp + timedelta(minutes=horizon_minutes)
            seasonal_factor = self.seasonal_patterns["hourly"][future_time.hour]
            
            # Simple linear prediction with seasonal adjustment
            trend_component = recent_trend * horizon_minutes
            predicted_load = current_load * (1 + trend_component) * seasonal_factor
            
            # Calculate confidence based on trend stability
            trend_variance = statistics.variance(self.trend_weights) if len(self.trend_weights) > 5 else 1.0
            confidence = max(0.1, min(1.0, 1.0 / (1.0 + trend_variance)))
            
            return max(0.0, predicted_load), confidence
            
        except Exception as e:
            self.logger.error(f"Error predicting load: {e}")
            return 0.0, 0.0
    
    def get_forecasting_stats(self) -> Dict[str, Any]:
        """Get forecasting statistics."""
        return {
            "history_size": len(self.metrics_history),
            "trend_stability": statistics.variance(self.trend_weights) if len(self.trend_weights) > 2 else 0.0,
            "seasonal_patterns": self.seasonal_patterns,
            "current_trend": statistics.mean(self.trend_weights) if self.trend_weights else 0.0
        }


class IntelligentAutoScaler:
    """Advanced auto-scaler with predictive capabilities and ML-based optimization."""
    
    def __init__(self, strategy: ScalingStrategy = ScalingStrategy.HYBRID):
        self.logger = logging.getLogger("intelligent_autoscaler")
        self.strategy = strategy
        
        # Rules and configuration
        self.scaling_rules: List[AdvancedScalingRule] = []
        self.metrics_history: deque = deque(maxlen=1440)  # 24 hours
        self.scaling_history: List[Dict[str, Any]] = []
        
        # Current state
        self.current_instances = 1
        self.target_instances = 1
        self.last_scaling_action: Optional[datetime] = None
        self.warmup_end_time: Optional[datetime] = None
        
        # Advanced components
        self.load_forecaster = PredictiveLoadForecaster()
        self.cost_optimizer = self._initialize_cost_optimizer()
        
        # Callbacks and monitoring
        self.scaling_callbacks: Dict[ScalingAction, List[Callable]] = {
            action: [] for action in ScalingAction
        }
        
        # Performance tracking
        self.scaling_efficiency: Dict[str, float] = {
            "prediction_accuracy": 0.0,
            "cost_savings": 0.0,
            "response_time_improvement": 0.0
        }
        
        # Thread pool for async operations
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Initialize with default rules
        self._setup_advanced_default_rules()
    
    def _initialize_cost_optimizer(self) -> Dict[str, Any]:
        """Initialize cost optimization module."""
        return {
            "enabled": True,
            "cost_per_instance_hour": 0.1,  # Default cost
            "performance_cost_ratio": 0.7,  # Weight performance vs cost
            "target_utilization": 75.0,     # Target CPU utilization for cost efficiency
        }
    
    def _setup_advanced_default_rules(self):
        """Setup advanced default scaling rules."""
        self.scaling_rules = [
            # Reactive CPU-based scaling
            AdvancedScalingRule(
                name="reactive_cpu_scale_up",
                priority=1,
                conditions=[
                    {"metric": "cpu_utilization", "operator": ">=", "threshold": 80.0},
                    {"metric": "response_time_p95", "operator": ">=", "threshold": 2000.0}
                ],
                action=ScalingAction.SCALE_UP,
                scale_factor=1.5,
                cooldown_seconds=180,
                performance_optimization=True
            ),
            
            # Predictive scaling based on load forecast
            AdvancedScalingRule(
                name="predictive_scale_up",
                priority=2,
                conditions=[
                    {"metric": "predicted_load_5m", "operator": ">=", "threshold": 80.0},
                    {"metric": "cpu_utilization", "operator": ">=", "threshold": 60.0}
                ],
                action=ScalingAction.SCALE_UP,
                scale_factor=1.3,
                cooldown_seconds=240,
                prediction_horizon=300,
                confidence_threshold=0.7
            ),
            
            # Memory pressure scaling
            AdvancedScalingRule(
                name="memory_pressure_scale_up",
                priority=3,
                conditions=[
                    {"metric": "memory_utilization", "operator": ">=", "threshold": 85.0},
                    {"metric": "memory_pressure", "operator": ">=", "threshold": 0.8}
                ],
                action=ScalingAction.SCALE_UP,
                scale_factor=1.4,
                cooldown_seconds=300
            ),
            
            # Queue-based horizontal scaling
            AdvancedScalingRule(
                name="queue_scale_out",
                priority=4,
                conditions=[
                    {"metric": "queue_length", "operator": ">=", "threshold": 50},
                    {"metric": "pending_evaluations", "operator": ">=", "threshold": 100}
                ],
                action=ScalingAction.SCALE_OUT,
                scale_factor=2.0,
                cooldown_seconds=120,
                max_instances=20
            ),
            
            # Cost-optimized scale down
            AdvancedScalingRule(
                name="cost_optimized_scale_down",
                priority=5,
                conditions=[
                    {"metric": "cpu_utilization", "operator": "<=", "threshold": 30.0},
                    {"metric": "memory_utilization", "operator": "<=", "threshold": 40.0},
                    {"metric": "queue_length", "operator": "<=", "threshold": 5}
                ],
                action=ScalingAction.SCALE_DOWN,
                scale_factor=0.7,
                cooldown_seconds=600,  # Longer cooldown for scale down
                cost_optimization=True
            )
        ]


class AutoScaler(IntelligentAutoScaler):
    """Backward compatible auto-scaler interface."""
    
    def _setup_default_rules(self):
        """Legacy method for backward compatibility."""
        self._setup_advanced_default_rules()
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules.append(rule)
        logger.info(f"Added scaling rule: {rule.name}")
    
    def register_scaling_callback(self, action: ScalingAction, callback: Callable):
        """Register a callback to execute when scaling occurs."""
        self.scaling_callbacks[action].append(callback)
        logger.info(f"Registered callback for {action.value}")
    
    async def update_metrics(self, metrics: AdvancedScalingMetrics):
        """Update metrics and trigger intelligent scaling evaluation."""
        # Add to history
        self.metrics_history.append(metrics)
        
        # Update load forecaster
        self.load_forecaster.add_metrics(metrics)
        
        # Add predictive metrics
        if self.strategy in [ScalingStrategy.PREDICTIVE, ScalingStrategy.HYBRID, ScalingStrategy.ML_BASED]:
            predicted_5m, confidence_5m = self.load_forecaster.predict_load(5)
            predicted_15m, confidence_15m = self.load_forecaster.predict_load(15)
            
            metrics.predicted_load_5m = predicted_5m
            metrics.predicted_load_15m = predicted_15m
        
        # Calculate load trend
        if len(self.metrics_history) >= 5:
            recent_loads = [self._calculate_composite_load(m) for m in list(self.metrics_history)[-5:]]
            metrics.load_trend = self._calculate_trend(recent_loads)
        
        # Skip scaling if in warmup period
        if self.warmup_end_time and datetime.now() < self.warmup_end_time:
            self.logger.debug("Skipping scaling decision during warmup period")
            return
        
        # Evaluate scaling decision
        action, rule_name = await self._evaluate_intelligent_scaling_decision(metrics)
        if action != ScalingAction.NO_ACTION:
            await self._execute_advanced_scaling_action(action, metrics, rule_name)
    
    def _calculate_composite_load(self, metrics: AdvancedScalingMetrics) -> float:
        """Calculate a composite load score."""
        return (
            0.4 * metrics.cpu_utilization +
            0.3 * metrics.memory_utilization +
            0.2 * min(100, metrics.queue_length * 2) +
            0.1 * min(100, metrics.response_time_p95 / 50)
        )
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression."""
        if len(values) < 2:
            return 0.0
        
        try:
            x = list(range(len(values)))
            n = len(values)
            sum_x = sum(x)
            sum_y = sum(values)
            sum_xy = sum(x[i] * values[i] for i in range(n))
            sum_x2 = sum(x[i] ** 2 for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
            return slope
        except (ZeroDivisionError, ValueError):
            return 0.0
    
    async def _evaluate_intelligent_scaling_decision(self, metrics: AdvancedScalingMetrics) -> Tuple[ScalingAction, str]:
        """Evaluate scaling decision using advanced rules and strategies."""
        if not self._can_scale():
            return ScalingAction.NO_ACTION, "cooldown_period"
        
        # Sort rules by priority
        sorted_rules = sorted(self.scaling_rules, key=lambda r: r.priority)
        
        for rule in sorted_rules:
            if rule.evaluate_conditions(metrics):
                # Additional validation for predictive rules
                if "predictive" in rule.name and rule.confidence_threshold > 0:
                    _, confidence = self.load_forecaster.predict_load(rule.prediction_horizon // 60)
                    if confidence < rule.confidence_threshold:
                        continue
                
                # Cost optimization check
                if rule.cost_optimization and self.cost_optimizer["enabled"]:
                    if not await self._is_cost_effective_scaling(rule.action, metrics):
                        continue
                
                return rule.action, rule.name
        
        # No rules triggered
        return ScalingAction.NO_ACTION, "no_rules_triggered"
    
    async def _is_cost_effective_scaling(self, action: ScalingAction, metrics: AdvancedScalingMetrics) -> bool:
        """Check if scaling action is cost-effective."""
        try:
            current_cost = self.current_instances * self.cost_optimizer["cost_per_instance_hour"]
            
            if action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_OUT]:
                new_instances = int(self.current_instances * 1.5)  # Estimated
                new_cost = new_instances * self.cost_optimizer["cost_per_instance_hour"]
                cost_increase = new_cost - current_cost
                
                # Check if performance improvement justifies cost
                performance_gain = (100 - metrics.cpu_utilization) / 100  # Simplified
                cost_effectiveness = performance_gain / max(cost_increase, 0.01)
                
                return cost_effectiveness > self.cost_optimizer["performance_cost_ratio"]
            
            return True  # Scale down is generally cost-effective
            
        except Exception as e:
            self.logger.error(f"Error in cost-effectiveness calculation: {e}")
            return True  # Default to allowing scaling
    
    async def _execute_advanced_scaling_action(self, action: ScalingAction, metrics: AdvancedScalingMetrics, rule_name: str):
        """Execute advanced scaling action with proper coordination."""
        try:
            old_instances = self.current_instances
            
            # Calculate new target instances
            if action == ScalingAction.SCALE_UP:
                self.target_instances = min(
                    int(self.current_instances * 1.5),
                    max(rule.max_instances for rule in self.scaling_rules if rule.name == rule_name)
                )
            elif action == ScalingAction.SCALE_DOWN:
                self.target_instances = max(
                    int(self.current_instances * 0.7),
                    min(rule.min_instances for rule in self.scaling_rules if rule.name == rule_name)
                )
            elif action == ScalingAction.SCALE_OUT:
                self.target_instances = min(
                    self.current_instances + 2,
                    max(rule.max_instances for rule in self.scaling_rules if rule.name == rule_name)
                )
            elif action == ScalingAction.SCALE_IN:
                self.target_instances = max(
                    self.current_instances - 1,
                    min(rule.min_instances for rule in self.scaling_rules if rule.name == rule_name)
                )
            
            if self.target_instances != self.current_instances:
                await self._scale_to_target(action, rule_name, metrics)
                
        except Exception as e:
            self.logger.error(f"Error executing scaling action {action}: {e}")
    
    async def _scale_to_target(self, action: ScalingAction, rule_name: str, metrics: AdvancedScalingMetrics):
        """Scale to target instances with proper coordination."""
        old_instances = self.current_instances
        self.current_instances = self.target_instances
        self.last_scaling_action = datetime.now()
        
        # Set warmup period for scale up actions
        if action in [ScalingAction.SCALE_UP, ScalingAction.SCALE_OUT]:
            warmup_seconds = next(
                (rule.warmup_time for rule in self.scaling_rules if rule.name == rule_name),
                60
            )
            self.warmup_end_time = datetime.now() + timedelta(seconds=warmup_seconds)
        
        # Record scaling event
        scaling_event = {
            "timestamp": self.last_scaling_action.isoformat(),
            "action": action.value,
            "rule_name": rule_name,
            "old_instances": old_instances,
            "new_instances": self.current_instances,
            "trigger_metrics": {
                "cpu_utilization": metrics.cpu_utilization,
                "memory_utilization": metrics.memory_utilization,
                "queue_length": metrics.queue_length,
                "response_time_p95": metrics.response_time_p95
            }
        }
        self.scaling_history.append(scaling_event)
        
        self.logger.info(
            f"Scaling {action.value}: {old_instances} -> {self.current_instances} instances "
            f"(rule: {rule_name}, warmup: {self.warmup_end_time})"
        )
        
        # Execute callbacks
        for callback in self.scaling_callbacks[action]:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(old_instances, self.current_instances, scaling_event)
                else:
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor, callback, old_instances, self.current_instances, scaling_event
                    )
            except Exception as e:
                self.logger.error(f"Error executing scaling callback: {e}")
    
    def _evaluate_scaling_decision(self, current_metrics: ScalingMetrics) -> ScalingAction:
        """Evaluate whether scaling is needed based on current metrics."""
        if not self._can_scale():
            return ScalingAction.NO_ACTION
        
        scale_up_votes = 0
        scale_down_votes = 0
        
        for rule in self.scaling_rules:
            metric_value = getattr(current_metrics, rule.metric_name, 0)
            
            if metric_value >= rule.threshold_up and self.current_instances < rule.max_instances:
                scale_up_votes += 1
                logger.debug(f"Rule {rule.name} votes for scale up: {metric_value} >= {rule.threshold_up}")
            elif metric_value <= rule.threshold_down and self.current_instances > rule.min_instances:
                scale_down_votes += 1
                logger.debug(f"Rule {rule.name} votes for scale down: {metric_value} <= {rule.threshold_down}")
        
        # Require majority vote for scaling action
        total_rules = len(self.scaling_rules)
        if scale_up_votes > total_rules // 2:
            return ScalingAction.SCALE_UP
        elif scale_down_votes > total_rules // 2:
            return ScalingAction.SCALE_DOWN
        else:
            return ScalingAction.NO_ACTION
    
    def _can_scale(self) -> bool:
        """Check if we can perform scaling (cooldown period)."""
        if self.last_scaling_action is None:
            return True
        
        # Use minimum cooldown from all rules
        min_cooldown = min(rule.cooldown_seconds for rule in self.scaling_rules)
        time_since_last = (datetime.now() - self.last_scaling_action).total_seconds()
        
        return time_since_last >= min_cooldown
    
    def _execute_scaling_action(self, action: ScalingAction, metrics: ScalingMetrics):
        """Execute the scaling action."""
        if action == ScalingAction.SCALE_UP:
            # Find the rule that triggered scale up
            triggered_rule = None
            for rule in self.scaling_rules:
                metric_value = getattr(metrics, rule.metric_name, 0)
                if metric_value >= rule.threshold_up:
                    triggered_rule = rule
                    break
            
            if triggered_rule:
                new_instances = min(
                    int(self.current_instances * triggered_rule.scale_factor),
                    triggered_rule.max_instances
                )
                self._scale_to(new_instances, action, triggered_rule.name)
        
        elif action == ScalingAction.SCALE_DOWN:
            # Find the rule that triggered scale down
            triggered_rule = None
            for rule in self.scaling_rules:
                metric_value = getattr(metrics, rule.metric_name, 0)
                if metric_value <= rule.threshold_down:
                    triggered_rule = rule
                    break
            
            if triggered_rule:
                new_instances = max(
                    int(self.current_instances / triggered_rule.scale_factor),
                    triggered_rule.min_instances
                )
                self._scale_to(new_instances, action, triggered_rule.name)
    
    def _scale_to(self, target_instances: int, action: ScalingAction, rule_name: str):
        """Scale to target number of instances."""
        if target_instances == self.current_instances:
            return
        
        old_instances = self.current_instances
        self.current_instances = target_instances
        self.last_scaling_action = datetime.now()
        
        logger.info(
            f"Scaling {action.value}: {old_instances} -> {target_instances} instances "
            f"(triggered by {rule_name})"
        )
        
        # Execute callbacks
        for callback in self.scaling_callbacks[action]:
            try:
                callback(old_instances, target_instances)
            except Exception as e:
                logger.error(f"Error executing scaling callback: {e}")
    
    def get_comprehensive_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive scaling status with advanced metrics."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        # Calculate scaling efficiency metrics
        recent_scaling_events = [
            event for event in self.scaling_history
            if datetime.fromisoformat(event["timestamp"]) > datetime.now() - timedelta(hours=24)
        ]
        
        return {
            "current_state": {
                "current_instances": self.current_instances,
                "target_instances": self.target_instances,
                "strategy": self.strategy.value,
                "warmup_active": self.warmup_end_time and datetime.now() < self.warmup_end_time,
                "warmup_end": self.warmup_end_time.isoformat() if self.warmup_end_time else None
            },
            "scaling_capability": {
                "can_scale": self._can_scale(),
                "last_scaling_action": self.last_scaling_action.isoformat() if self.last_scaling_action else None,
                "scaling_rules_count": len(self.scaling_rules),
                "active_rules": len([r for r in self.scaling_rules if r.enabled])
            },
            "metrics_status": {
                "history_size": len(self.metrics_history),
                "latest_timestamp": latest_metrics.timestamp.isoformat() if latest_metrics else None,
                "load_trend": latest_metrics.load_trend if latest_metrics else 0.0,
                "seasonal_factor": latest_metrics.seasonal_factor if latest_metrics else 1.0
            },
            "latest_metrics": {
                "cpu_utilization": latest_metrics.cpu_utilization if latest_metrics else 0,
                "memory_utilization": latest_metrics.memory_utilization if latest_metrics else 0,
                "queue_length": latest_metrics.queue_length if latest_metrics else 0,
                "response_time_p95": latest_metrics.response_time_p95 if latest_metrics else 0,
                "predicted_load_5m": latest_metrics.predicted_load_5m if latest_metrics else 0,
                "predicted_load_15m": latest_metrics.predicted_load_15m if latest_metrics else 0
            } if latest_metrics else None,
            "efficiency_metrics": {
                "scaling_events_24h": len(recent_scaling_events),
                "prediction_accuracy": self.scaling_efficiency["prediction_accuracy"],
                "cost_savings": self.scaling_efficiency["cost_savings"],
                "response_time_improvement": self.scaling_efficiency["response_time_improvement"]
            },
            "forecasting_stats": self.load_forecaster.get_forecasting_stats(),
            "cost_optimization": self.cost_optimizer
        }
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Legacy method for backward compatibility."""
        comprehensive_status = self.get_comprehensive_scaling_status()
        
        # Return simplified status for backward compatibility
        return {
            "current_instances": comprehensive_status["current_state"]["current_instances"],
            "last_scaling_action": comprehensive_status["scaling_capability"]["last_scaling_action"],
            "can_scale": comprehensive_status["scaling_capability"]["can_scale"],
            "scaling_rules_count": comprehensive_status["scaling_capability"]["scaling_rules_count"],
            "metrics_history_size": comprehensive_status["metrics_status"]["history_size"],
            "latest_metrics": comprehensive_status["latest_metrics"]
        }
    
    def get_scaling_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get scaling decisions and metrics history."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            {
                "timestamp": metrics.timestamp.isoformat(),
                "cpu_utilization": metrics.cpu_utilization,
                "memory_utilization": metrics.memory_utilization,
                "active_requests": metrics.active_requests,
                "queue_length": metrics.queue_length,
                "response_time_p95": metrics.response_time_p95,
                "error_rate": metrics.error_rate,
                "throughput_qps": metrics.throughput_qps
            }
            for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def simulate_load_test(self, duration_seconds: int = 300):
        """Simulate various load conditions for testing auto-scaling."""
        logger.info(f"Starting auto-scaling simulation for {duration_seconds} seconds")
        
        start_time = time.time()
        while time.time() - start_time < duration_seconds:
            # Simulate varying load conditions
            elapsed = time.time() - start_time
            progress = elapsed / duration_seconds
            
            # Create varying load patterns
            if progress < 0.2:  # Low load
                cpu = 20 + (progress * 50)
                memory = 30 + (progress * 20)
                queue = 5
                response_time = 300
            elif progress < 0.5:  # Increasing load
                cpu = 70 + ((progress - 0.2) * 100)
                memory = 50 + ((progress - 0.2) * 150)
                queue = 10 + ((progress - 0.2) * 200)
                response_time = 500 + ((progress - 0.2) * 2000)
            elif progress < 0.8:  # High load
                cpu = 90 + (progress * 10)
                memory = 85 + (progress * 10)
                queue = 60 + (progress * 40)
                response_time = 1500 + (progress * 1000)
            else:  # Decreasing load
                cpu = 100 - ((progress - 0.8) * 200)
                memory = 95 - ((progress - 0.8) * 150)
                queue = 100 - ((progress - 0.8) * 300)
                response_time = 2500 - ((progress - 0.8) * 2000)
            
            # Create and update metrics
            metrics = AdvancedScalingMetrics(
                cpu_utilization=max(0, min(100, cpu)),
                memory_utilization=max(0, min(100, memory)),
                active_requests=int(queue * 0.8),
                queue_length=max(0, int(queue)),
                response_time_p95=max(100, response_time),
                error_rate=max(0, (cpu - 80) * 0.1) if cpu > 80 else 0,
                throughput_qps=max(1, 100 - (response_time / 50))
            )
            
            asyncio.run(self.update_metrics(metrics)) if hasattr(self, 'update_metrics') and asyncio.iscoroutinefunction(self.update_metrics) else self.update_metrics(metrics)
            time.sleep(10)  # Update every 10 seconds
        
        logger.info("Auto-scaling simulation completed")


class LoadBalancer:
    """Intelligent load balancer with health checks."""
    
    def __init__(self):
        self.workers: List[Dict[str, Any]] = []
        self.health_check_interval = 30  # seconds
        self.unhealthy_threshold = 3  # consecutive failures
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
    
    def add_worker(self, worker_id: str, endpoint: str, weight: float = 1.0):
        """Add a worker to the load balancer."""
        worker = {
            "id": worker_id,
            "endpoint": endpoint,
            "weight": weight,
            "healthy": True,
            "consecutive_failures": 0,
            "total_requests": 0,
            "total_errors": 0,
            "avg_response_time": 0.0,
            "last_health_check": None
        }
        self.workers.append(worker)
        logger.info(f"Added worker {worker_id} at {endpoint}")
    
    def remove_worker(self, worker_id: str):
        """Remove a worker from the load balancer."""
        self.workers = [w for w in self.workers if w["id"] != worker_id]
        logger.info(f"Removed worker {worker_id}")
    
    def get_next_worker(self, strategy: str = "weighted_round_robin") -> Optional[Dict[str, Any]]:
        """Get the next worker based on load balancing strategy."""
        healthy_workers = [w for w in self.workers if w["healthy"]]
        
        if not healthy_workers:
            logger.error("No healthy workers available")
            return None
        
        if strategy == "round_robin":
            return min(healthy_workers, key=lambda w: w["total_requests"])
        
        elif strategy == "weighted_round_robin":
            # Consider both weight and current load
            best_worker = None
            best_score = float('inf')
            
            for worker in healthy_workers:
                # Score = requests / weight (lower is better)
                score = worker["total_requests"] / worker["weight"]
                if score < best_score:
                    best_score = score
                    best_worker = worker
            
            return best_worker
        
        elif strategy == "least_response_time":
            return min(healthy_workers, key=lambda w: w["avg_response_time"])
        
        elif strategy == "least_errors":
            return min(healthy_workers, key=lambda w: w["total_errors"])
        
        else:
            # Default to first healthy worker
            return healthy_workers[0]
    
    def record_request(self, worker_id: str, response_time: float, success: bool):
        """Record a request result for a worker."""
        for worker in self.workers:
            if worker["id"] == worker_id:
                worker["total_requests"] += 1
                
                if success:
                    worker["consecutive_failures"] = 0
                    # Update average response time (exponential moving average)
                    alpha = 0.1
                    worker["avg_response_time"] = (
                        alpha * response_time + 
                        (1 - alpha) * worker["avg_response_time"]
                    )
                else:
                    worker["total_errors"] += 1
                    worker["consecutive_failures"] += 1
                    
                    # Mark as unhealthy if too many consecutive failures
                    if worker["consecutive_failures"] >= self.unhealthy_threshold:
                        worker["healthy"] = False
                        logger.warning(f"Worker {worker_id} marked as unhealthy")
                
                break
    
    async def start_health_checks(self):
        """Start periodic health checks for all workers."""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Started health checks for load balancer")
    
    async def stop_health_checks(self):
        """Stop health checks."""
        if not self._running:
            return
        
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped health checks for load balancer")
    
    async def _health_check_loop(self):
        """Main health check loop."""
        while self._running:
            for worker in self.workers:
                try:
                    # Simulate health check (in real implementation, make HTTP request)
                    is_healthy = await self._check_worker_health(worker)
                    
                    if is_healthy and not worker["healthy"]:
                        worker["healthy"] = True
                        worker["consecutive_failures"] = 0
                        logger.info(f"Worker {worker['id']} recovered and marked healthy")
                    elif not is_healthy:
                        worker["consecutive_failures"] += 1
                        if worker["consecutive_failures"] >= self.unhealthy_threshold:
                            worker["healthy"] = False
                    
                    worker["last_health_check"] = datetime.now()
                    
                except Exception as e:
                    logger.error(f"Health check failed for worker {worker['id']}: {e}")
                    worker["consecutive_failures"] += 1
            
            await asyncio.sleep(self.health_check_interval)
    
    async def _check_worker_health(self, worker: Dict[str, Any]) -> bool:
        """Check if a worker is healthy (placeholder implementation)."""
        # In real implementation, this would make an HTTP request to worker's health endpoint
        # For now, simulate based on error rate
        error_rate = worker["total_errors"] / max(1, worker["total_requests"])
        return error_rate < 0.1  # Healthy if error rate < 10%
    
    def get_load_balancer_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        total_workers = len(self.workers)
        healthy_workers = len([w for w in self.workers if w["healthy"]])
        total_requests = sum(w["total_requests"] for w in self.workers)
        total_errors = sum(w["total_errors"] for w in self.workers)
        
        return {
            "total_workers": total_workers,
            "healthy_workers": healthy_workers,
            "unhealthy_workers": total_workers - healthy_workers,
            "total_requests": total_requests,
            "total_errors": total_errors,
            "overall_error_rate": (total_errors / max(1, total_requests)) * 100,
            "workers": [
                {
                    "id": w["id"],
                    "endpoint": w["endpoint"],
                    "healthy": w["healthy"],
                    "weight": w["weight"],
                    "total_requests": w["total_requests"],
                    "total_errors": w["total_errors"],
                    "error_rate": (w["total_errors"] / max(1, w["total_requests"])) * 100,
                    "avg_response_time": w["avg_response_time"],
                    "last_health_check": w["last_health_check"].isoformat() if w["last_health_check"] else None
                }
                for w in self.workers
            ]
        }


    def optimize_scaling_strategy(self) -> Dict[str, Any]:
        """Optimize scaling strategy based on historical performance."""
        if len(self.scaling_history) < 10:
            return {"message": "Not enough scaling history for optimization"}
        
        try:
            # Analyze scaling patterns
            scaling_effectiveness = self._analyze_scaling_effectiveness()
            rule_performance = self._analyze_rule_performance()
            cost_analysis = self._analyze_cost_efficiency()
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(
                scaling_effectiveness, rule_performance, cost_analysis
            )
            
            return {
                "scaling_effectiveness": scaling_effectiveness,
                "rule_performance": rule_performance,
                "cost_analysis": cost_analysis,
                "recommendations": recommendations
            }
            
        except Exception as e:
            self.logger.error(f"Error optimizing scaling strategy: {e}")
            return {"error": str(e)}
    
    def _analyze_scaling_effectiveness(self) -> Dict[str, float]:
        """Analyze how effective scaling actions have been."""
        if len(self.scaling_history) < 5:
            return {}
        
        # Calculate metrics like scaling accuracy, response time improvement, etc.
        scale_up_events = [e for e in self.scaling_history if "up" in e["action"] or "out" in e["action"]]
        scale_down_events = [e for e in self.scaling_history if "down" in e["action"] or "in" in e["action"]]
        
        return {
            "total_scaling_events": len(self.scaling_history),
            "scale_up_ratio": len(scale_up_events) / len(self.scaling_history),
            "scale_down_ratio": len(scale_down_events) / len(self.scaling_history),
            "avg_time_between_scaling": self._calculate_avg_scaling_interval()
        }
    
    def _analyze_rule_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance of individual scaling rules."""
        rule_stats = defaultdict(lambda: {"triggers": 0, "effectiveness": 0.0})
        
        for event in self.scaling_history:
            rule_name = event.get("rule_name", "unknown")
            rule_stats[rule_name]["triggers"] += 1
        
        return dict(rule_stats)
    
    def _analyze_cost_efficiency(self) -> Dict[str, float]:
        """Analyze cost efficiency of scaling decisions."""
        if not self.cost_optimizer["enabled"]:
            return {"cost_optimization_disabled": True}
        
        # Calculate cost metrics
        total_instance_hours = sum(
            event["new_instances"] for event in self.scaling_history
        )
        
        return {
            "total_instance_hours": total_instance_hours,
            "estimated_cost": total_instance_hours * self.cost_optimizer["cost_per_instance_hour"],
            "cost_per_request": 0.0  # Would be calculated with actual request data
        }
    
    def _generate_optimization_recommendations(self, effectiveness, rule_performance, cost_analysis) -> List[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # Check scaling frequency
        if effectiveness.get("avg_time_between_scaling", 600) < 300:
            recommendations.append("Consider increasing cooldown periods to reduce scaling frequency")
        
        # Check rule efficiency
        for rule_name, stats in rule_performance.items():
            if stats["triggers"] > len(self.scaling_history) * 0.5:
                recommendations.append(f"Rule '{rule_name}' triggers very frequently - consider adjusting thresholds")
        
        # Cost optimization
        if cost_analysis.get("estimated_cost", 0) > 100:  # Arbitrary threshold
            recommendations.append("Consider enabling more aggressive cost optimization")
        
        return recommendations
    
    def _calculate_avg_scaling_interval(self) -> float:
        """Calculate average time between scaling events."""
        if len(self.scaling_history) < 2:
            return 0.0
        
        intervals = []
        for i in range(1, len(self.scaling_history)):
            prev_time = datetime.fromisoformat(self.scaling_history[i-1]["timestamp"])
            curr_time = datetime.fromisoformat(self.scaling_history[i]["timestamp"])
            intervals.append((curr_time - prev_time).total_seconds())
        
        return statistics.mean(intervals) if intervals else 0.0


# Enhanced global instances
intelligent_auto_scaler = IntelligentAutoScaler()
load_balancer = LoadBalancer()

# Legacy compatibility
auto_scaler = AutoScaler()