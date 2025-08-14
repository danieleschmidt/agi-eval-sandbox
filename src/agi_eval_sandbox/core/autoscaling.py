"""Auto-scaling and resource management for AGI Evaluation Sandbox."""

import asyncio
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json

from .logging_config import get_logger
from .exceptions import ResourceError

logger = get_logger("autoscaling")


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


@dataclass
class ScalingMetrics:
    """Metrics used for scaling decisions."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    active_requests: int = 0
    queue_length: int = 0
    response_time_p95: float = 0.0
    error_rate: float = 0.0
    throughput_qps: float = 0.0


@dataclass
class ScalingRule:
    """Defines when and how to scale."""
    name: str
    metric_name: str  # "cpu_utilization", "memory_utilization", "queue_length", etc.
    threshold_up: float
    threshold_down: float
    cooldown_seconds: int = 300  # Minimum time between scaling actions
    scale_factor: float = 1.5  # How much to scale by
    max_instances: int = 10
    min_instances: int = 1


class AutoScaler:
    """Intelligent auto-scaling based on multiple metrics."""
    
    def __init__(self):
        self.scaling_rules: List[ScalingRule] = []
        self.metrics_history: List[ScalingMetrics] = []
        self.last_scaling_action: Optional[datetime] = None
        self.current_instances = 1
        self.scaling_callbacks: Dict[ScalingAction, List[Callable]] = {
            ScalingAction.SCALE_UP: [],
            ScalingAction.SCALE_DOWN: []
        }
        
        # Initialize default scaling rules
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default scaling rules."""
        self.scaling_rules = [
            ScalingRule(
                name="cpu_based_scaling",
                metric_name="cpu_utilization",
                threshold_up=80.0,
                threshold_down=30.0,
                cooldown_seconds=300,
                scale_factor=1.5,
                max_instances=10
            ),
            ScalingRule(
                name="memory_based_scaling",
                metric_name="memory_utilization",
                threshold_up=85.0,
                threshold_down=40.0,
                cooldown_seconds=300,
                scale_factor=1.3,
                max_instances=10
            ),
            ScalingRule(
                name="queue_based_scaling",
                metric_name="queue_length",
                threshold_up=50.0,
                threshold_down=10.0,
                cooldown_seconds=180,
                scale_factor=2.0,
                max_instances=15
            ),
            ScalingRule(
                name="response_time_scaling",
                metric_name="response_time_p95",
                threshold_up=2000.0,  # 2 seconds
                threshold_down=500.0,  # 0.5 seconds
                cooldown_seconds=240,
                scale_factor=1.8,
                max_instances=12
            )
        ]
    
    def add_scaling_rule(self, rule: ScalingRule):
        """Add a custom scaling rule."""
        self.scaling_rules.append(rule)
        logger.info(f"Added scaling rule: {rule.name}")
    
    def register_scaling_callback(self, action: ScalingAction, callback: Callable):
        """Register a callback to execute when scaling occurs."""
        self.scaling_callbacks[action].append(callback)
        logger.info(f"Registered callback for {action.value}")
    
    def update_metrics(self, metrics: ScalingMetrics):
        """Update metrics and trigger scaling evaluation."""
        self.metrics_history.append(metrics)
        
        # Keep only last hour of metrics
        cutoff_time = datetime.now() - timedelta(hours=1)
        self.metrics_history = [
            m for m in self.metrics_history if m.timestamp > cutoff_time
        ]
        
        # Evaluate scaling decision
        action = self._evaluate_scaling_decision(metrics)
        if action != ScalingAction.NO_ACTION:
            self._execute_scaling_action(action, metrics)
    
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
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        
        return {
            "current_instances": self.current_instances,
            "last_scaling_action": self.last_scaling_action.isoformat() if self.last_scaling_action else None,
            "can_scale": self._can_scale(),
            "scaling_rules_count": len(self.scaling_rules),
            "metrics_history_size": len(self.metrics_history),
            "latest_metrics": {
                "cpu_utilization": latest_metrics.cpu_utilization if latest_metrics else 0,
                "memory_utilization": latest_metrics.memory_utilization if latest_metrics else 0,
                "active_requests": latest_metrics.active_requests if latest_metrics else 0,
                "queue_length": latest_metrics.queue_length if latest_metrics else 0,
                "response_time_p95": latest_metrics.response_time_p95 if latest_metrics else 0,
            } if latest_metrics else None
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
            metrics = ScalingMetrics(
                cpu_utilization=max(0, min(100, cpu)),
                memory_utilization=max(0, min(100, memory)),
                active_requests=int(queue * 0.8),
                queue_length=max(0, int(queue)),
                response_time_p95=max(100, response_time),
                error_rate=max(0, (cpu - 80) * 0.1) if cpu > 80 else 0,
                throughput_qps=max(1, 100 - (response_time / 50))
            )
            
            self.update_metrics(metrics)
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


# Global instances
auto_scaler = AutoScaler()
load_balancer = LoadBalancer()