"""Health check and system monitoring utilities."""

try:
    import psutil
except ImportError:
    psutil = None

import asyncio
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum

from .exceptions import ResourceError
from .logging_config import get_logger

logger = get_logger("health")


class HealthStatus(Enum):
    """Health check status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Individual health check result."""
    name: str
    status: HealthStatus
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    duration_ms: float = 0.0


@dataclass
class SystemMetrics:
    """System resource metrics."""
    cpu_percent: float
    memory_percent: float
    memory_available_mb: float
    disk_usage_percent: float
    disk_free_mb: float
    load_average: List[float]
    timestamp: datetime = field(default_factory=datetime.now)


class HealthMonitor:
    """Comprehensive health monitoring system."""
    
    def __init__(self):
        self.checks: Dict[str, HealthCheck] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 100
        
        # Thresholds
        self.cpu_warning_threshold = 70.0
        self.cpu_critical_threshold = 90.0
        self.memory_warning_threshold = 80.0
        self.memory_critical_threshold = 95.0
        self.disk_warning_threshold = 80.0
        self.disk_critical_threshold = 95.0
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks and return results."""
        checks = [
            self.check_system_resources(),
            self.check_memory_usage(),
            self.check_disk_space(),
            self.check_process_health(),
        ]
        
        results = await asyncio.gather(*checks, return_exceptions=True)
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                check_name = f"check_{i}"
                self.checks[check_name] = HealthCheck(
                    name=check_name,
                    status=HealthStatus.CRITICAL,
                    message=f"Health check failed: {str(result)}",
                    details={"exception": str(result)}
                )
            elif isinstance(result, HealthCheck):
                self.checks[result.name] = result
        
        return self.checks
    
    async def check_system_resources(self) -> HealthCheck:
        """Check overall system resource usage."""
        start_time = time.time()
        
        if psutil is None:
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.UNKNOWN,
                message="psutil not available - system monitoring disabled",
                duration_ms=(time.time() - start_time) * 1000
            )
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)  # Reduced interval for testing
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on thresholds
            status = HealthStatus.HEALTHY
            issues = []
            
            if cpu_percent >= self.cpu_critical_threshold:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent >= self.cpu_warning_threshold:
                status = HealthStatus.WARNING
                issues.append(f"CPU usage warning: {cpu_percent:.1f}%")
            
            if memory.percent >= self.memory_critical_threshold:
                status = HealthStatus.CRITICAL
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent >= self.memory_warning_threshold:
                if status != HealthStatus.CRITICAL:
                    status = HealthStatus.WARNING
                issues.append(f"Memory usage warning: {memory.percent:.1f}%")
            
            if disk.percent >= self.disk_critical_threshold:
                status = HealthStatus.CRITICAL
                issues.append(f"Disk usage critical: {disk.percent:.1f}%")
            elif disk.percent >= self.disk_warning_threshold:
                if status != HealthStatus.CRITICAL:
                    status = HealthStatus.WARNING
                issues.append(f"Disk usage warning: {disk.percent:.1f}%")
            
            message = "System resources normal"
            if issues:
                message = "; ".join(issues)
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="system_resources",
                status=status,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_mb": memory.available / (1024 * 1024),
                    "disk_percent": disk.percent,
                    "disk_free_mb": disk.free / (1024 * 1024)
                },
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"System resource check failed: {e}")
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check system resources: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def check_memory_usage(self) -> HealthCheck:
        """Check memory usage and detect leaks."""
        start_time = time.time()
        
        if psutil is None:
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.UNKNOWN,
                message="psutil not available - memory monitoring disabled",
                duration_ms=(time.time() - start_time) * 1000
            )
        
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # Check for memory leaks (simplified)
            status = HealthStatus.HEALTHY
            message = "Memory usage normal"
            
            if memory_percent > 50.0:  # Process using > 50% of system memory
                status = HealthStatus.WARNING
                message = f"High memory usage: {memory_percent:.1f}%"
            
            if memory_percent > 80.0:  # Process using > 80% of system memory
                status = HealthStatus.CRITICAL
                message = f"Critical memory usage: {memory_percent:.1f}%"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="memory_usage",
                status=status,
                message=message,
                details={
                    "rss_mb": memory_info.rss / (1024 * 1024),
                    "vms_mb": memory_info.vms / (1024 * 1024),
                    "percent": memory_percent
                },
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Memory usage check failed: {e}")
            return HealthCheck(
                name="memory_usage",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check memory usage: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def check_disk_space(self) -> HealthCheck:
        """Check available disk space."""
        start_time = time.time()
        
        if psutil is None:
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message="psutil not available - disk monitoring disabled",
                duration_ms=(time.time() - start_time) * 1000
            )
        
        try:
            disk_usage = psutil.disk_usage('/')
            free_gb = disk_usage.free / (1024 ** 3)
            
            status = HealthStatus.HEALTHY
            message = f"Disk space available: {free_gb:.1f}GB"
            
            if free_gb < 1.0:  # Less than 1GB free
                status = HealthStatus.CRITICAL
                message = f"Critical: Only {free_gb:.1f}GB disk space remaining"
            elif free_gb < 5.0:  # Less than 5GB free
                status = HealthStatus.WARNING
                message = f"Warning: Only {free_gb:.1f}GB disk space remaining"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="disk_space",
                status=status,
                message=message,
                details={
                    "total_gb": disk_usage.total / (1024 ** 3),
                    "used_gb": disk_usage.used / (1024 ** 3),
                    "free_gb": free_gb,
                    "percent_used": (disk_usage.used / disk_usage.total) * 100
                },
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Disk space check failed: {e}")
            return HealthCheck(
                name="disk_space",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check disk space: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    async def check_process_health(self) -> HealthCheck:
        """Check current process health."""
        start_time = time.time()
        
        if psutil is None:
            return HealthCheck(
                name="process_health",
                status=HealthStatus.UNKNOWN,
                message="psutil not available - process monitoring disabled",
                duration_ms=(time.time() - start_time) * 1000
            )
        
        try:
            process = psutil.Process()
            
            # Check if process is responsive
            status = HealthStatus.HEALTHY
            message = "Process is healthy"
            
            # Check file descriptors (Unix systems)
            try:
                num_fds = process.num_fds()
                if num_fds > 1000:  # Arbitrary threshold
                    status = HealthStatus.WARNING
                    message = f"High number of file descriptors: {num_fds}"
            except (AttributeError, psutil.AccessDenied):
                # num_fds() not available on Windows
                num_fds = 0
            
            # Check threads
            num_threads = process.num_threads()
            if num_threads > 50:  # Arbitrary threshold
                status = HealthStatus.WARNING
                message = f"High number of threads: {num_threads}"
            
            duration_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name="process_health",
                status=status,
                message=message,
                details={
                    "pid": process.pid,
                    "num_threads": num_threads,
                    "num_fds": num_fds,
                    "status": process.status(),
                    "create_time": datetime.fromtimestamp(process.create_time()).isoformat()
                },
                duration_ms=duration_ms
            )
            
        except Exception as e:
            logger.error(f"Process health check failed: {e}")
            return HealthCheck(
                name="process_health",
                status=HealthStatus.CRITICAL,
                message=f"Failed to check process health: {str(e)}",
                duration_ms=(time.time() - start_time) * 1000
            )
    
    def collect_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        if psutil is None:
            # Return mock metrics when psutil is not available
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                memory_available_mb=1024.0,
                disk_usage_percent=0.0,
                disk_free_mb=10240.0,
                load_average=[0.0, 0.0, 0.0]
            )
        
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0]
            
            metrics = SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_available_mb=memory.available / (1024 * 1024),
                disk_usage_percent=disk.percent,
                disk_free_mb=disk.free / (1024 * 1024),
                load_average=list(load_avg)
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to collect metrics: {e}")
            raise ResourceError(f"Failed to collect system metrics: {str(e)}")
    
    def get_overall_health(self) -> HealthStatus:
        """Get overall system health status."""
        if not self.checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in self.checks.values()]
        
        if HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.WARNING in statuses:
            return HealthStatus.WARNING
        elif all(status == HealthStatus.HEALTHY for status in statuses):
            return HealthStatus.HEALTHY
        else:
            return HealthStatus.UNKNOWN
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        overall_status = self.get_overall_health()
        
        return {
            "overall_status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "duration_ms": check.duration_ms,
                    "details": check.details
                }
                for name, check in self.checks.items()
            },
            "metrics": self.metrics_history[-1].__dict__ if self.metrics_history else None
        }


# Global health monitor instance
health_monitor = HealthMonitor()