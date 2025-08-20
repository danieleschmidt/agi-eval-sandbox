"""Enhanced monitoring and metrics collection for AGI Evaluation Sandbox - Generation 2 Robust Implementation."""

import time
import asyncio
import psutil
import logging
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
import json
import statistics
from collections import deque, defaultdict
import warnings

from .logging_config import get_logger, performance_logger
from .exceptions import ResourceError

logger = get_logger("monitoring")


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):
    """Types of metrics."""
    SYSTEM = "system"
    EVALUATION = "evaluation"
    SECURITY = "security"
    BUSINESS = "business"
    CUSTOM = "custom"


@dataclass
class Alert:
    """System alert with detailed information."""
    alert_id: str
    timestamp: datetime
    level: AlertLevel
    metric_type: MetricType
    metric_name: str
    current_value: Union[float, int, str]
    threshold_value: Union[float, int, str]
    message: str
    actions_taken: List[str] = field(default_factory=list)
    resolved: bool = False
    resolution_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level.value,
            "metric_type": self.metric_type.value,
            "metric_name": self.metric_name,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "message": self.message,
            "actions_taken": self.actions_taken,
            "resolved": self.resolved,
            "resolution_time": self.resolution_time.isoformat() if self.resolution_time else None
        }


@dataclass
class SystemMetrics:
    """Enhanced system resource metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    cpu_cores: int = 0
    cpu_frequency_mhz: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    memory_total_mb: float = 0.0
    disk_usage_percent: float = 0.0
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    network_packets_sent: int = 0
    network_packets_recv: int = 0
    active_connections: int = 0
    load_average_1m: float = 0.0
    load_average_5m: float = 0.0
    load_average_15m: float = 0.0
    temperature_celsius: Optional[float] = None
    gpu_usage_percent: Optional[float] = None
    gpu_memory_percent: Optional[float] = None


@dataclass
class EvaluationMetrics:
    """Enhanced evaluation performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    total_evaluations: int = 0
    active_evaluations: int = 0
    completed_evaluations: int = 0
    failed_evaluations: int = 0
    avg_response_time_ms: float = 0.0
    min_response_time_ms: float = 0.0
    max_response_time_ms: float = 0.0
    p95_response_time_ms: float = 0.0
    p99_response_time_ms: float = 0.0
    throughput_qps: float = 0.0
    success_rate: float = 100.0
    error_rate: float = 0.0
    timeout_rate: float = 0.0
    retry_rate: float = 0.0
    cache_hit_rate: float = 0.0
    cache_miss_rate: float = 0.0
    concurrent_limit_reached: int = 0
    queue_depth: int = 0
    model_provider_breakdown: Dict[str, int] = field(default_factory=dict)
    benchmark_breakdown: Dict[str, int] = field(default_factory=dict)
    error_breakdown: Dict[str, int] = field(default_factory=dict)


@dataclass
class SecurityMetrics:
    """Security monitoring metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    total_requests: int = 0
    blocked_requests: int = 0
    suspicious_requests: int = 0
    failed_authentications: int = 0
    rate_limited_requests: int = 0
    threats_detected: int = 0
    threats_by_type: Dict[str, int] = field(default_factory=dict)
    unique_threat_ips: int = 0
    blocked_ips: int = 0
    false_positive_rate: float = 0.0
    avg_threat_score: float = 0.0
    compliance_violations: int = 0


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self):
        self.logger = logging.getLogger("alert_manager")
        self.alerts: List[Alert] = []
        self.alert_thresholds = self._default_thresholds()
        self.alert_callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self.alert_suppression: Dict[str, datetime] = {}  # metric_name -> suppress_until
        self.suppression_duration = timedelta(minutes=5)  # Default suppression
    
    def _default_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Default alert thresholds."""
        return {
            "cpu_percent": {"warning": 80.0, "critical": 95.0},
            "memory_percent": {"warning": 85.0, "critical": 95.0},
            "disk_usage_percent": {"warning": 80.0, "critical": 90.0},
            "error_rate": {"warning": 5.0, "critical": 10.0},
            "response_time_ms": {"warning": 5000.0, "critical": 10000.0},
            "success_rate": {"warning": 95.0, "critical": 90.0},
            "threats_detected": {"warning": 10, "critical": 50},
            "blocked_requests": {"warning": 100, "critical": 500}
        }
    
    def register_alert_callback(self, metric_name: str, callback: Callable[[Alert], None]):
        """Register callback for specific metric alerts."""
        self.alert_callbacks[metric_name].append(callback)
    
    def check_thresholds(
        self,
        metrics: Dict[str, Union[SystemMetrics, EvaluationMetrics, SecurityMetrics]]
    ) -> List[Alert]:
        """Check metrics against thresholds and generate alerts."""
        new_alerts = []
        
        for metric_type, metric_data in metrics.items():
            if isinstance(metric_data, SystemMetrics):
                new_alerts.extend(self._check_system_thresholds(metric_data))
            elif isinstance(metric_data, EvaluationMetrics):
                new_alerts.extend(self._check_evaluation_thresholds(metric_data))
            elif isinstance(metric_data, SecurityMetrics):
                new_alerts.extend(self._check_security_thresholds(metric_data))
        
        # Process new alerts
        for alert in new_alerts:
            self._process_alert(alert)
        
        return new_alerts
    
    def _check_system_thresholds(self, metrics: SystemMetrics) -> List[Alert]:
        """Check system metrics against thresholds."""
        alerts = []
        
        checks = [
            ("cpu_percent", metrics.cpu_percent, MetricType.SYSTEM),
            ("memory_percent", metrics.memory_percent, MetricType.SYSTEM),
            ("disk_usage_percent", metrics.disk_usage_percent, MetricType.SYSTEM)
        ]
        
        for metric_name, current_value, metric_type in checks:
            alert = self._check_metric_threshold(
                metric_name, current_value, metric_type, metrics.timestamp
            )
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _check_evaluation_thresholds(self, metrics: EvaluationMetrics) -> List[Alert]:
        """Check evaluation metrics against thresholds."""
        alerts = []
        
        checks = [
            ("error_rate", metrics.error_rate, MetricType.EVALUATION),
            ("response_time_ms", metrics.avg_response_time_ms, MetricType.EVALUATION)
        ]
        
        # Success rate check (inverted - alert when below threshold)
        if "success_rate" in self.alert_thresholds:
            thresholds = self.alert_thresholds["success_rate"]
            if metrics.success_rate <= thresholds.get("critical", 90.0):
                level = AlertLevel.CRITICAL
            elif metrics.success_rate <= thresholds.get("warning", 95.0):
                level = AlertLevel.WARNING
            else:
                level = None
            
            if level and not self._is_suppressed("success_rate"):
                alert = Alert(
                    alert_id=f"success_rate_{int(time.time())}",
                    timestamp=metrics.timestamp,
                    level=level,
                    metric_type=MetricType.EVALUATION,
                    metric_name="success_rate",
                    current_value=metrics.success_rate,
                    threshold_value=thresholds.get(level.value, 95.0),
                    message=f"Success rate {metrics.success_rate:.1f}% below threshold"
                )
                alerts.append(alert)
        
        for metric_name, current_value, metric_type in checks:
            alert = self._check_metric_threshold(
                metric_name, current_value, metric_type, metrics.timestamp
            )
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _check_security_thresholds(self, metrics: SecurityMetrics) -> List[Alert]:
        """Check security metrics against thresholds."""
        alerts = []
        
        checks = [
            ("threats_detected", metrics.threats_detected, MetricType.SECURITY),
            ("blocked_requests", metrics.blocked_requests, MetricType.SECURITY)
        ]
        
        for metric_name, current_value, metric_type in checks:
            alert = self._check_metric_threshold(
                metric_name, current_value, metric_type, metrics.timestamp
            )
            if alert:
                alerts.append(alert)
        
        return alerts
    
    def _check_metric_threshold(
        self,
        metric_name: str,
        current_value: Union[float, int],
        metric_type: MetricType,
        timestamp: datetime
    ) -> Optional[Alert]:
        """Check individual metric against thresholds."""
        if metric_name not in self.alert_thresholds or self._is_suppressed(metric_name):
            return None
        
        thresholds = self.alert_thresholds[metric_name]
        
        if current_value >= thresholds.get("critical", float('inf')):
            level = AlertLevel.CRITICAL
        elif current_value >= thresholds.get("warning", float('inf')):
            level = AlertLevel.WARNING
        else:
            return None
        
        return Alert(
            alert_id=f"{metric_name}_{int(time.time())}",
            timestamp=timestamp,
            level=level,
            metric_type=metric_type,
            metric_name=metric_name,
            current_value=current_value,
            threshold_value=thresholds.get(level.value, 0),
            message=f"{metric_name} {current_value} exceeds {level.value} threshold"
        )
    
    def _is_suppressed(self, metric_name: str) -> bool:
        """Check if alerts for this metric are suppressed."""
        if metric_name in self.alert_suppression:
            if datetime.now() < self.alert_suppression[metric_name]:
                return True
            else:
                del self.alert_suppression[metric_name]
        return False
    
    def _process_alert(self, alert: Alert):
        """Process new alert - store, log, and trigger callbacks."""
        self.alerts.append(alert)
        
        # Log alert
        log_level = {
            AlertLevel.INFO: logging.INFO,
            AlertLevel.WARNING: logging.WARNING,
            AlertLevel.ERROR: logging.ERROR,
            AlertLevel.CRITICAL: logging.CRITICAL
        }.get(alert.level, logging.WARNING)
        
        self.logger.log(log_level, f"Alert: {alert.message}", extra=alert.to_dict())
        
        # Suppress similar alerts
        self.alert_suppression[alert.metric_name] = datetime.now() + self.suppression_duration
        
        # Trigger callbacks
        for callback in self.alert_callbacks.get(alert.metric_name, []):
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def resolve_alert(self, alert_id: str, resolution_message: str = ""):
        """Mark alert as resolved."""
        for alert in self.alerts:
            if alert.alert_id == alert_id and not alert.resolved:
                alert.resolved = True
                alert.resolution_time = datetime.now()
                if resolution_message:
                    alert.actions_taken.append(resolution_message)
                self.logger.info(f"Alert {alert_id} resolved: {resolution_message}")
                break
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all unresolved alerts."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get alert summary for specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_alerts = [a for a in self.alerts if a.timestamp > cutoff_time]
        
        level_counts = defaultdict(int)
        type_counts = defaultdict(int)
        
        for alert in recent_alerts:
            level_counts[alert.level.value] += 1
            type_counts[alert.metric_type.value] += 1
        
        return {
            "time_period_hours": hours,
            "total_alerts": len(recent_alerts),
            "active_alerts": len(self.get_active_alerts()),
            "alert_levels": dict(level_counts),
            "alert_types": dict(type_counts),
            "mean_time_to_resolution": self._calculate_mttr(recent_alerts)
        }
    
    def _calculate_mttr(self, alerts: List[Alert]) -> float:
        """Calculate mean time to resolution for resolved alerts."""
        resolved_alerts = [a for a in alerts if a.resolved and a.resolution_time]
        if not resolved_alerts:
            return 0.0
        
        resolution_times = [
            (a.resolution_time - a.timestamp).total_seconds() / 60  # minutes
            for a in resolved_alerts
        ]
        
        return statistics.mean(resolution_times)


class AdvancedMetricsCollector:
    """Advanced metrics collection with real-time analytics and alerting."""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.logger = logging.getLogger("metrics_collector")
        
        # Enhanced metrics storage
        self.system_metrics_history: deque = deque(maxlen=1440)  # 24 hours at 1min intervals
        self.evaluation_metrics_history: deque = deque(maxlen=1440)
        self.security_metrics_history: deque = deque(maxlen=1440)
        
        # Real-time response times for percentile calculations
        self.response_times_window: deque = deque(maxlen=1000)
        
        # Collection state
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Enhanced counters
        self.total_evaluations = 0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        self.timeout_evaluations = 0
        self.retry_evaluations = 0
        self.total_response_time = 0.0
        
        # Security counters
        self.total_requests = 0
        self.blocked_requests = 0
        self.suspicious_requests = 0
        self.threats_detected = 0
        
        # Alert management
        self.alert_manager = AlertManager()
        
        # Custom metrics
        self.custom_metrics: Dict[str, Any] = {}
        
        # Performance optimization
        self._metrics_cache: Dict[str, Any] = {}
        self._cache_expiry = datetime.now()
        self._cache_duration = timedelta(seconds=30)


class MetricsCollector(AdvancedMetricsCollector):
    """Backward compatible metrics collector interface."""
    
    async def start_collection(self):
        """Start metrics collection."""
        if self._running:
            logger.warning("Metrics collection already running")
            return
        
        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info(f"Started metrics collection with {self.collection_interval}s interval")
    
    async def stop_collection(self):
        """Stop metrics collection."""
        if not self._running:
            return
        
        self._running = False
        if self._collection_task:
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped metrics collection")
    
    async def _enhanced_collection_loop(self):
        """Enhanced collection loop with alerting and analytics."""
        while self._running:
            try:
                # Collect all metrics
                system_metrics = await self._collect_enhanced_system_metrics()
                eval_metrics = self._collect_enhanced_evaluation_metrics()
                security_metrics = self._collect_security_metrics()
                
                # Store metrics
                self.system_metrics_history.append(system_metrics)
                self.evaluation_metrics_history.append(eval_metrics)
                self.security_metrics_history.append(security_metrics)
                
                # Check thresholds and generate alerts
                metrics_dict = {
                    "system": system_metrics,
                    "evaluation": eval_metrics,
                    "security": security_metrics
                }
                
                new_alerts = self.alert_manager.check_thresholds(metrics_dict)
                if new_alerts:
                    self.logger.info(f"Generated {len(new_alerts)} new alerts")
                
                # Log enhanced metrics
                performance_logger.log_system_metrics(
                    cpu_percent=system_metrics.cpu_percent,
                    memory_percent=system_metrics.memory_percent,
                    disk_usage_percent=system_metrics.disk_usage_percent
                )
                
                # Clear metrics cache
                self._invalidate_cache()
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error in enhanced metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collection_loop(self):
        """Legacy collection loop for backward compatibility."""
        await self._enhanced_collection_loop()
    
    async def _collect_enhanced_system_metrics(self) -> SystemMetrics:
        """Collect comprehensive system resource metrics."""
        try:
            # CPU information
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # CPU frequency (if available)
            cpu_freq = 0.0
            try:
                freq_info = psutil.cpu_freq()
                cpu_freq = freq_info.current if freq_info else 0.0
            except (AttributeError, OSError):
                pass
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            memory_total_mb = memory.total / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            disk_free_gb = disk.free / (1024 * 1024 * 1024)
            disk_total_gb = disk.total / (1024 * 1024 * 1024)
            
            # Network usage
            network = psutil.net_io_counters()
            
            # Load averages (Unix-like systems)
            load_1m = load_5m = load_15m = 0.0
            try:
                load_avg = psutil.getloadavg()
                load_1m, load_5m, load_15m = load_avg
            except (AttributeError, OSError):
                pass
            
            # Temperature (if available)
            temperature = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    # Get first available temperature sensor
                    for sensor_list in temps.values():
                        if sensor_list:
                            temperature = sensor_list[0].current
                            break
            except (AttributeError, OSError):
                pass
            
            # GPU metrics (if available)
            gpu_usage = gpu_memory = None
            try:
                # This would require additional GPU libraries like pynvml
                # For now, we'll leave as placeholder
                pass
            except Exception:
                pass
            
            # Active connections
            connections = 0
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, OSError):
                pass
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                cpu_cores=cpu_count,
                cpu_frequency_mhz=cpu_freq,
                memory_percent=memory.percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                memory_total_mb=memory_total_mb,
                disk_usage_percent=disk_usage_percent,
                disk_free_gb=disk_free_gb,
                disk_total_gb=disk_total_gb,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                network_packets_sent=network.packets_sent,
                network_packets_recv=network.packets_recv,
                active_connections=connections,
                load_average_1m=load_1m,
                load_average_5m=load_5m,
                load_average_15m=load_15m,
                temperature_celsius=temperature,
                gpu_usage_percent=gpu_usage,
                gpu_memory_percent=gpu_memory
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics()
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Legacy system metrics collection for backward compatibility."""
        enhanced_metrics = await self._collect_enhanced_system_metrics()
        
        # Return simplified metrics for backward compatibility
        return SystemMetrics(
            timestamp=enhanced_metrics.timestamp,
            cpu_percent=enhanced_metrics.cpu_percent,
            memory_percent=enhanced_metrics.memory_percent,
            memory_used_mb=enhanced_metrics.memory_used_mb,
            memory_available_mb=enhanced_metrics.memory_available_mb,
            disk_usage_percent=enhanced_metrics.disk_usage_percent,
            network_bytes_sent=enhanced_metrics.network_bytes_sent,
            network_bytes_recv=enhanced_metrics.network_bytes_recv,
            active_connections=enhanced_metrics.active_connections
        )
    
    def _collect_enhanced_evaluation_metrics(self) -> EvaluationMetrics:
        """Collect comprehensive evaluation performance metrics."""
        try:
            # Calculate rates
            success_rate = (
                (self.successful_evaluations / self.total_evaluations * 100)
                if self.total_evaluations > 0 else 100.0
            )
            
            error_rate = (
                (self.failed_evaluations / self.total_evaluations * 100)
                if self.total_evaluations > 0 else 0.0
            )
            
            timeout_rate = (
                (self.timeout_evaluations / self.total_evaluations * 100)
                if self.total_evaluations > 0 else 0.0
            )
            
            retry_rate = (
                (self.retry_evaluations / self.total_evaluations * 100)
                if self.total_evaluations > 0 else 0.0
            )
            
            # Response time statistics
            avg_response_time_ms = (
                (self.total_response_time / self.total_evaluations * 1000)
                if self.total_evaluations > 0 else 0.0
            )
            
            # Calculate percentiles from recent response times
            min_time = max_time = p95_time = p99_time = 0.0
            if self.response_times_window:
                sorted_times = sorted(self.response_times_window)
                min_time = min(sorted_times) * 1000  # Convert to ms
                max_time = max(sorted_times) * 1000
                
                if len(sorted_times) >= 20:  # Need enough samples for percentiles
                    p95_idx = int(len(sorted_times) * 0.95)
                    p99_idx = int(len(sorted_times) * 0.99)
                    p95_time = sorted_times[p95_idx] * 1000
                    p99_time = sorted_times[p99_idx] * 1000
            
            # Calculate throughput (evaluations per second over collection interval)
            throughput_qps = self.total_evaluations / max(self.collection_interval, 1)
            
            return EvaluationMetrics(
                total_evaluations=self.total_evaluations,
                completed_evaluations=self.successful_evaluations,
                failed_evaluations=self.failed_evaluations,
                active_evaluations=0,  # Would be set by evaluator
                avg_response_time_ms=avg_response_time_ms,
                min_response_time_ms=min_time,
                max_response_time_ms=max_time,
                p95_response_time_ms=p95_time,
                p99_response_time_ms=p99_time,
                throughput_qps=throughput_qps,
                success_rate=success_rate,
                error_rate=error_rate,
                timeout_rate=timeout_rate,
                retry_rate=retry_rate,
                cache_hit_rate=0.0,  # Would be set by cache manager
                cache_miss_rate=0.0,
                concurrent_limit_reached=0,
                queue_depth=0,
                model_provider_breakdown={},
                benchmark_breakdown={},
                error_breakdown={}
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting evaluation metrics: {e}")
            return EvaluationMetrics()
    
    def _collect_evaluation_metrics(self) -> EvaluationMetrics:
        """Legacy evaluation metrics collection for backward compatibility."""
        enhanced_metrics = self._collect_enhanced_evaluation_metrics()
        
        # Return simplified metrics for backward compatibility
        return EvaluationMetrics(
            timestamp=enhanced_metrics.timestamp,
            total_evaluations=enhanced_metrics.total_evaluations,
            active_evaluations=enhanced_metrics.active_evaluations,
            avg_response_time_ms=enhanced_metrics.avg_response_time_ms,
            throughput_qps=enhanced_metrics.throughput_qps,
            success_rate=enhanced_metrics.success_rate,
            error_rate=enhanced_metrics.error_rate,
            cache_hit_rate=enhanced_metrics.cache_hit_rate
        )
    
    def _collect_security_metrics(self) -> SecurityMetrics:
        """Collect security monitoring metrics."""
        try:
            # Calculate security rates
            blocked_rate = (
                (self.blocked_requests / self.total_requests * 100)
                if self.total_requests > 0 else 0.0
            )
            
            suspicious_rate = (
                (self.suspicious_requests / self.total_requests * 100)
                if self.total_requests > 0 else 0.0
            )
            
            return SecurityMetrics(
                total_requests=self.total_requests,
                blocked_requests=self.blocked_requests,
                suspicious_requests=self.suspicious_requests,
                threats_detected=self.threats_detected,
                blocked_ips=0,  # Would be set by security monitor
                false_positive_rate=0.0,  # Would be calculated from labeled data
                avg_threat_score=0.0
            )
            
        except Exception as e:
            self.logger.error(f"Error collecting security metrics: {e}")
            return SecurityMetrics()
    
    def _trim_history(self):
        """Trim metrics history - deques automatically handle max length."""
        # Deques automatically maintain max length, but we can add cleanup for old data
        pass
    
    def record_evaluation_start(self):
        """Record the start of an evaluation."""
        self.total_evaluations += 1
    
    def record_evaluation_success(self, duration_seconds: float):
        """Record a successful evaluation."""
        self.successful_evaluations += 1
        self.total_response_time += duration_seconds
        
        # Store for percentile calculations
        self.response_times_window.append(duration_seconds)
    
    def record_evaluation_timeout(self):
        """Record an evaluation timeout."""
        self.timeout_evaluations += 1
        self.failed_evaluations += 1
    
    def record_evaluation_retry(self):
        """Record an evaluation retry."""
        self.retry_evaluations += 1
    
    def record_security_event(self, event_type: str):
        """Record a security event."""
        if event_type == "blocked_request":
            self.blocked_requests += 1
        elif event_type == "suspicious_request":
            self.suspicious_requests += 1
        elif event_type == "threat_detected":
            self.threats_detected += 1
        
        self.total_requests += 1
    
    def add_custom_metric(self, name: str, value: Union[int, float, str], tags: Optional[Dict[str, str]] = None):
        """Add custom metric."""
        self.custom_metrics[name] = {
            "value": value,
            "timestamp": datetime.now(),
            "tags": tags or {}
        }
    
    def _invalidate_cache(self):
        """Invalidate metrics cache."""
        self._cache_expiry = datetime.now()
        self._metrics_cache.clear()
    
    def record_evaluation_failure(self):
        """Record a failed evaluation."""
        self.failed_evaluations += 1
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest collected metrics."""
        latest_system = self.system_metrics_history[-1] if self.system_metrics_history else SystemMetrics()
        latest_eval = self.evaluation_metrics_history[-1] if self.evaluation_metrics_history else EvaluationMetrics()
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": latest_system.cpu_percent,
                "memory_percent": latest_system.memory_percent,
                "memory_used_mb": latest_system.memory_used_mb,
                "memory_available_mb": latest_system.memory_available_mb,
                "disk_usage_percent": latest_system.disk_usage_percent,
                "active_connections": latest_system.active_connections
            },
            "evaluation": {
                "total_evaluations": latest_eval.total_evaluations,
                "active_evaluations": latest_eval.active_evaluations,
                "avg_response_time_ms": latest_eval.avg_response_time_ms,
                "throughput_qps": latest_eval.throughput_qps,
                "success_rate": latest_eval.success_rate,
                "error_rate": latest_eval.error_rate,
                "cache_hit_rate": latest_eval.cache_hit_rate
            }
        }
    
    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get metrics summary over specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter metrics within time window
        recent_system = [
            m for m in self.system_metrics_history
            if m.timestamp > cutoff_time
        ]
        
        recent_eval = [
            m for m in self.evaluation_metrics_history
            if m.timestamp > cutoff_time
        ]
        
        if not recent_system or not recent_eval:
            return self.get_latest_metrics()
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in recent_system) / len(recent_system)
        avg_memory = sum(m.memory_percent for m in recent_system) / len(recent_system)
        avg_response_time = sum(m.avg_response_time_ms for m in recent_eval) / len(recent_eval)
        avg_throughput = sum(m.throughput_qps for m in recent_eval) / len(recent_eval)
        avg_success_rate = sum(m.success_rate for m in recent_eval) / len(recent_eval)
        
        return {
            "time_period_hours": hours,
            "summary": {
                "avg_cpu_percent": round(avg_cpu, 2),
                "avg_memory_percent": round(avg_memory, 2),
                "avg_response_time_ms": round(avg_response_time, 2),
                "avg_throughput_qps": round(avg_throughput, 2),
                "avg_success_rate": round(avg_success_rate, 2),
                "total_data_points": len(recent_system)
            }
        }
    
    def export_metrics(self, file_path: str):
        """Export metrics to JSON file."""
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "system_metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "memory_used_mb": m.memory_used_mb,
                    "disk_usage_percent": m.disk_usage_percent,
                    "active_connections": m.active_connections
                }
                for m in self.system_metrics_history
            ],
            "evaluation_metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "total_evaluations": m.total_evaluations,
                    "avg_response_time_ms": m.avg_response_time_ms,
                    "throughput_qps": m.throughput_qps,
                    "success_rate": m.success_rate,
                    "error_rate": m.error_rate
                }
                for m in self.evaluation_metrics_history
            ]
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported metrics to {file_path}")


    def get_real_time_dashboard_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data with caching."""
        now = datetime.now()
        
        # Check cache
        if now < self._cache_expiry and self._metrics_cache:
            return self._metrics_cache
        
        # Collect latest metrics
        latest_system = self.system_metrics_history[-1] if self.system_metrics_history else SystemMetrics()
        latest_eval = self.evaluation_metrics_history[-1] if self.evaluation_metrics_history else EvaluationMetrics()
        latest_security = self.security_metrics_history[-1] if self.security_metrics_history else SecurityMetrics()
        
        # Get active alerts
        active_alerts = self.alert_manager.get_active_alerts()
        
        dashboard_data = {
            "timestamp": now.isoformat(),
            "system_health": {
                "status": self._calculate_system_health_status(latest_system),
                "cpu_percent": latest_system.cpu_percent,
                "memory_percent": latest_system.memory_percent,
                "disk_usage_percent": latest_system.disk_usage_percent,
                "load_average": latest_system.load_average_1m,
                "temperature": latest_system.temperature_celsius
            },
            "evaluation_performance": {
                "status": self._calculate_eval_health_status(latest_eval),
                "success_rate": latest_eval.success_rate,
                "avg_response_time_ms": latest_eval.avg_response_time_ms,
                "throughput_qps": latest_eval.throughput_qps,
                "active_evaluations": latest_eval.active_evaluations,
                "total_evaluations": latest_eval.total_evaluations
            },
            "security_status": {
                "status": self._calculate_security_health_status(latest_security),
                "threats_detected": latest_security.threats_detected,
                "blocked_requests": latest_security.blocked_requests,
                "total_requests": latest_security.total_requests
            },
            "alerts": {
                "active_count": len(active_alerts),
                "critical_count": len([a for a in active_alerts if a.level == AlertLevel.CRITICAL]),
                "recent_alerts": [a.to_dict() for a in active_alerts[:5]]  # Last 5 alerts
            },
            "custom_metrics": self.custom_metrics
        }
        
        # Cache the data
        self._metrics_cache = dashboard_data
        self._cache_expiry = now + self._cache_duration
        
        return dashboard_data
    
    def _calculate_system_health_status(self, metrics: SystemMetrics) -> str:
        """Calculate overall system health status."""
        if (metrics.cpu_percent > 95 or 
            metrics.memory_percent > 95 or 
            metrics.disk_usage_percent > 90):
            return "critical"
        elif (metrics.cpu_percent > 80 or 
              metrics.memory_percent > 85 or 
              metrics.disk_usage_percent > 80):
            return "warning"
        return "healthy"
    
    def _calculate_eval_health_status(self, metrics: EvaluationMetrics) -> str:
        """Calculate evaluation health status."""
        if metrics.error_rate > 10 or metrics.success_rate < 90:
            return "critical"
        elif metrics.error_rate > 5 or metrics.success_rate < 95:
            return "warning"
        return "healthy"
    
    def _calculate_security_health_status(self, metrics: SecurityMetrics) -> str:
        """Calculate security health status."""
        if metrics.threats_detected > 50:
            return "critical"
        elif metrics.threats_detected > 10:
            return "warning"
        return "healthy"


# Enhanced global instances
metrics_collector = AdvancedMetricsCollector()
alert_manager = AlertManager()

# Legacy compatibility
legacy_metrics_collector = MetricsCollector()