#!/usr/bin/env python3
"""
Robust Health Monitor - Generation 2 Enhancement
Advanced health monitoring with predictive failure detection
"""

import asyncio
import time
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    DOWN = "down"

@dataclass
class HealthMetric:
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str = ""
    trend: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    
    @property
    def status(self) -> HealthStatus:
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY

@dataclass
class HealthCheck:
    name: str
    status: HealthStatus
    metrics: List[HealthMetric]
    message: str
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)

class RobustHealthMonitor:
    """Advanced health monitoring with predictive capabilities."""
    
    def __init__(self):
        self.logger = logging.getLogger("health_monitor")
        self.metrics_history: Dict[str, List[float]] = {}
        self.alert_callbacks: List[Callable] = []
        self.monitoring_enabled = True
        
        # Health thresholds
        self.thresholds = {
            "cpu_percent": {"warning": 80.0, "critical": 95.0},
            "memory_percent": {"warning": 85.0, "critical": 95.0},
            "disk_percent": {"warning": 85.0, "critical": 95.0},
            "load_average": {"warning": 4.0, "critical": 8.0},
            "response_time_ms": {"warning": 1000.0, "critical": 5000.0},
            "error_rate": {"warning": 5.0, "critical": 15.0},
            "active_connections": {"warning": 1000, "critical": 2000}
        }
    
    def add_alert_callback(self, callback: Callable[[HealthCheck], None]):
        """Add callback for health alerts."""
        self.alert_callbacks.append(callback)
    
    def record_metric(self, name: str, value: float):
        """Record metric for trend analysis."""
        if name not in self.metrics_history:
            self.metrics_history[name] = []
        
        self.metrics_history[name].append(value)
        
        # Keep only last 100 values
        if len(self.metrics_history[name]) > 100:
            self.metrics_history[name] = self.metrics_history[name][-100:]
    
    def calculate_trend(self, name: str) -> Optional[str]:
        """Calculate trend for a metric."""
        if name not in self.metrics_history or len(self.metrics_history[name]) < 3:
            return None
        
        values = self.metrics_history[name][-10:]  # Last 10 values
        if len(values) < 3:
            return None
        
        # Simple trend calculation
        recent_avg = sum(values[-3:]) / 3
        older_avg = sum(values[:-3]) / len(values[:-3])
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    async def check_system_resources(self) -> HealthCheck:
        """Check system resource utilization."""
        try:
            # CPU Usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric("cpu_percent", cpu_percent)
            
            # Memory Usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.record_metric("memory_percent", memory_percent)
            
            # Disk Usage
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            self.record_metric("disk_percent", disk_percent)
            
            # Load Average (Unix only)
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
                self.record_metric("load_average", load_avg)
            except AttributeError:
                load_avg = 0.0  # Windows doesn't have load average
            
            metrics = [
                HealthMetric(
                    name="cpu_usage",
                    value=cpu_percent,
                    threshold_warning=self.thresholds["cpu_percent"]["warning"],
                    threshold_critical=self.thresholds["cpu_percent"]["critical"],
                    unit="%",
                    trend=self.calculate_trend("cpu_percent")
                ),
                HealthMetric(
                    name="memory_usage",
                    value=memory_percent,
                    threshold_warning=self.thresholds["memory_percent"]["warning"],
                    threshold_critical=self.thresholds["memory_percent"]["critical"],
                    unit="%",
                    trend=self.calculate_trend("memory_percent")
                ),
                HealthMetric(
                    name="disk_usage",
                    value=disk_percent,
                    threshold_warning=self.thresholds["disk_percent"]["warning"],
                    threshold_critical=self.thresholds["disk_percent"]["critical"],
                    unit="%",
                    trend=self.calculate_trend("disk_percent")
                ),
                HealthMetric(
                    name="load_average",
                    value=load_avg,
                    threshold_warning=self.thresholds["load_average"]["warning"],
                    threshold_critical=self.thresholds["load_average"]["critical"],
                    unit="",
                    trend=self.calculate_trend("load_average")
                )
            ]
            
            # Determine overall status
            critical_metrics = [m for m in metrics if m.status == HealthStatus.CRITICAL]
            degraded_metrics = [m for m in metrics if m.status == HealthStatus.DEGRADED]
            
            if critical_metrics:
                status = HealthStatus.CRITICAL
                message = f"Critical resource usage detected: {[m.name for m in critical_metrics]}"
            elif degraded_metrics:
                status = HealthStatus.DEGRADED
                message = f"Degraded resource usage: {[m.name for m in degraded_metrics]}"
            else:
                status = HealthStatus.HEALTHY
                message = "All system resources within normal ranges"
            
            health_check = HealthCheck(
                name="system_resources",
                status=status,
                metrics=metrics,
                message=message,
                details={
                    "total_memory_gb": memory.total / (1024**3),
                    "available_memory_gb": memory.available / (1024**3),
                    "total_disk_gb": disk.total / (1024**3),
                    "free_disk_gb": disk.free / (1024**3),
                    "cpu_count": psutil.cpu_count()
                }
            )
            
            # Trigger alerts if needed
            if status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                await self._trigger_alerts(health_check)
            
            return health_check
            
        except Exception as e:
            self.logger.error(f"System resource check failed: {e}")
            return HealthCheck(
                name="system_resources",
                status=HealthStatus.DOWN,
                metrics=[],
                message=f"Health check failed: {str(e)}"
            )
    
    async def check_application_health(self) -> HealthCheck:
        """Check application-specific health metrics."""
        try:
            metrics = []
            
            # Check if core modules can be imported
            try:
                import sys
                sys.path.insert(0, str(Path(__file__).parent / "src"))
                from agi_eval_sandbox.core.evaluator import EvalSuite
                from agi_eval_sandbox.core.models import Model
                
                # Test basic functionality
                eval_suite = EvalSuite()
                benchmarks = eval_suite.list_benchmarks()
                
                metrics.append(HealthMetric(
                    name="available_benchmarks",
                    value=len(benchmarks),
                    threshold_warning=1,
                    threshold_critical=0,
                    unit="count"
                ))
                
                app_status = HealthStatus.HEALTHY
                message = f"Application healthy with {len(benchmarks)} benchmarks available"
                
            except Exception as import_error:
                metrics.append(HealthMetric(
                    name="core_imports",
                    value=0,
                    threshold_warning=0.5,
                    threshold_critical=0,
                    unit="status"
                ))
                
                app_status = HealthStatus.CRITICAL
                message = f"Core import failed: {str(import_error)}"
            
            # Check file system permissions
            try:
                test_file = Path(__file__).parent / "health_test.tmp"
                test_file.write_text("health check")
                content = test_file.read_text()
                test_file.unlink()
                
                if content == "health check":
                    file_system_ok = 1.0
                else:
                    file_system_ok = 0.0
                    
            except Exception:
                file_system_ok = 0.0
            
            metrics.append(HealthMetric(
                name="filesystem_access",
                value=file_system_ok,
                threshold_warning=0.5,
                threshold_critical=0,
                unit="status"
            ))
            
            # Overall application status
            critical_count = sum(1 for m in metrics if m.status == HealthStatus.CRITICAL)
            degraded_count = sum(1 for m in metrics if m.status == HealthStatus.DEGRADED)
            
            if critical_count > 0:
                final_status = HealthStatus.CRITICAL
                final_message = f"Application critical issues detected ({critical_count} critical)"
            elif degraded_count > 0:
                final_status = HealthStatus.DEGRADED
                final_message = f"Application degraded ({degraded_count} warnings)"
            else:
                final_status = HealthStatus.HEALTHY
                final_message = "Application healthy"
            
            health_check = HealthCheck(
                name="application_health",
                status=final_status,
                metrics=metrics,
                message=final_message,
                details={
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                    "working_directory": str(Path.cwd()),
                    "application_path": str(Path(__file__).parent)
                }
            )
            
            if final_status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                await self._trigger_alerts(health_check)
            
            return health_check
            
        except Exception as e:
            self.logger.error(f"Application health check failed: {e}")
            return HealthCheck(
                name="application_health",
                status=HealthStatus.DOWN,
                metrics=[],
                message=f"Application health check failed: {str(e)}"
            )
    
    async def check_dependencies(self) -> HealthCheck:
        """Check external dependencies status."""
        try:
            metrics = []
            
            # Check Python version
            import sys
            python_version = sys.version_info
            if python_version.major == 3 and python_version.minor >= 9:
                python_status = 1.0
            else:
                python_status = 0.0
            
            metrics.append(HealthMetric(
                name="python_version",
                value=python_status,
                threshold_warning=0.5,
                threshold_critical=0,
                unit="status"
            ))
            
            # Check core Python modules
            required_modules = [
                "asyncio", "json", "pathlib", "typing",
                "dataclasses", "enum", "time", "logging"
            ]
            
            available_modules = 0
            for module in required_modules:
                try:
                    __import__(module)
                    available_modules += 1
                except ImportError:
                    pass
            
            module_availability = available_modules / len(required_modules)
            metrics.append(HealthMetric(
                name="required_modules",
                value=module_availability * 100,
                threshold_warning=90.0,
                threshold_critical=80.0,
                unit="%"
            ))
            
            # Overall dependency status
            critical_deps = [m for m in metrics if m.status == HealthStatus.CRITICAL]
            degraded_deps = [m for m in metrics if m.status == HealthStatus.DEGRADED]
            
            if critical_deps:
                status = HealthStatus.CRITICAL
                message = f"Critical dependency issues: {[m.name for m in critical_deps]}"
            elif degraded_deps:
                status = HealthStatus.DEGRADED
                message = f"Dependency warnings: {[m.name for m in degraded_deps]}"
            else:
                status = HealthStatus.HEALTHY
                message = "All dependencies satisfied"
            
            health_check = HealthCheck(
                name="dependencies",
                status=status,
                metrics=metrics,
                message=message,
                details={
                    "python_version": f"{python_version.major}.{python_version.minor}.{python_version.micro}",
                    "available_modules": available_modules,
                    "required_modules": len(required_modules)
                }
            )
            
            if status in [HealthStatus.CRITICAL, HealthStatus.DEGRADED]:
                await self._trigger_alerts(health_check)
            
            return health_check
            
        except Exception as e:
            self.logger.error(f"Dependency check failed: {e}")
            return HealthCheck(
                name="dependencies",
                status=HealthStatus.DOWN,
                metrics=[],
                message=f"Dependency check failed: {str(e)}"
            )
    
    async def _trigger_alerts(self, health_check: HealthCheck):
        """Trigger alert callbacks for health issues."""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(health_check)
                else:
                    callback(health_check)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    async def run_comprehensive_health_check(self) -> Dict[str, HealthCheck]:
        """Run all health checks and return results."""
        if not self.monitoring_enabled:
            return {}
        
        checks = {}
        
        # Run all health checks in parallel
        check_tasks = [
            ("system_resources", self.check_system_resources()),
            ("application_health", self.check_application_health()),
            ("dependencies", self.check_dependencies())
        ]
        
        results = await asyncio.gather(*[task for _, task in check_tasks], return_exceptions=True)
        
        for (name, _), result in zip(check_tasks, results):
            if isinstance(result, Exception):
                checks[name] = HealthCheck(
                    name=name,
                    status=HealthStatus.DOWN,
                    metrics=[],
                    message=f"Health check crashed: {str(result)}"
                )
            else:
                checks[name] = result
        
        return checks
    
    def get_overall_status(self, checks: Dict[str, HealthCheck]) -> HealthStatus:
        """Determine overall system status from individual checks."""
        if not checks:
            return HealthStatus.DOWN
        
        statuses = [check.status for check in checks.values()]
        
        if HealthStatus.DOWN in statuses:
            return HealthStatus.DOWN
        elif HealthStatus.CRITICAL in statuses:
            return HealthStatus.CRITICAL
        elif HealthStatus.DEGRADED in statuses:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    def export_health_report(self, checks: Dict[str, HealthCheck]) -> Dict[str, Any]:
        """Export comprehensive health report."""
        overall_status = self.get_overall_status(checks)
        
        return {
            "timestamp": time.time(),
            "overall_status": overall_status.value,
            "checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "metrics": [
                        {
                            "name": metric.name,
                            "value": metric.value,
                            "unit": metric.unit,
                            "status": metric.status.value,
                            "trend": metric.trend,
                            "thresholds": {
                                "warning": metric.threshold_warning,
                                "critical": metric.threshold_critical
                            }
                        }
                        for metric in check.metrics
                    ],
                    "details": check.details,
                    "timestamp": check.timestamp
                }
                for name, check in checks.items()
            },
            "summary": {
                "total_checks": len(checks),
                "healthy": sum(1 for c in checks.values() if c.status == HealthStatus.HEALTHY),
                "degraded": sum(1 for c in checks.values() if c.status == HealthStatus.DEGRADED),
                "critical": sum(1 for c in checks.values() if c.status == HealthStatus.CRITICAL),
                "down": sum(1 for c in checks.values() if c.status == HealthStatus.DOWN)
            }
        }

async def main():
    """Run health monitoring example."""
    print("\n" + "="*60)
    print("🩺 ROBUST HEALTH MONITOR - GENERATION 2")
    print("="*60)
    
    monitor = RobustHealthMonitor()
    
    # Add simple alert callback
    def alert_callback(health_check: HealthCheck):
        print(f"🚨 ALERT: {health_check.name} - {health_check.status.value.upper()}")
        print(f"   Message: {health_check.message}")
    
    monitor.add_alert_callback(alert_callback)
    
    # Run comprehensive health check
    print("🔍 Running comprehensive health checks...")
    checks = await monitor.run_comprehensive_health_check()
    
    # Generate and display report
    report = monitor.export_health_report(checks)
    
    print(f"\n📊 HEALTH REPORT")
    print(f"Overall Status: {report['overall_status'].upper()}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(report['timestamp']))}")
    
    for check_name, check_data in report['checks'].items():
        status_emoji = {
            'healthy': '✅',
            'degraded': '⚠️',
            'critical': '🚨',
            'down': '❌'
        }.get(check_data['status'], '❓')
        
        print(f"\n{status_emoji} {check_name.upper()}: {check_data['status'].upper()}")
        print(f"   {check_data['message']}")
        
        if check_data['metrics']:
            print("   Metrics:")
            for metric in check_data['metrics']:
                trend_symbol = {
                    'increasing': '📈',
                    'decreasing': '📉',
                    'stable': '➖'
                }.get(metric.get('trend'), '')
                
                print(f"     • {metric['name']}: {metric['value']:.1f}{metric['unit']} "
                      f"({metric['status']}) {trend_symbol}")
    
    # Summary
    summary = report['summary']
    print(f"\n📈 SUMMARY:")
    print(f"   Total Checks: {summary['total_checks']}")
    print(f"   ✅ Healthy: {summary['healthy']}")
    print(f"   ⚠️ Degraded: {summary['degraded']}")
    print(f"   🚨 Critical: {summary['critical']}")
    print(f"   ❌ Down: {summary['down']}")
    
    overall_status = monitor.get_overall_status(checks)
    if overall_status == HealthStatus.HEALTHY:
        print("\n🎉 SYSTEM ROBUST AND HEALTHY!")
    elif overall_status == HealthStatus.DEGRADED:
        print("\n⚠️ SYSTEM DEGRADED - MONITORING REQUIRED")
    else:
        print("\n🚨 SYSTEM REQUIRES ATTENTION")
    
    print("="*60)

if __name__ == "__main__":
    asyncio.run(main())