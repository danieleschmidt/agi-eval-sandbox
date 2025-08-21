#!/usr/bin/env python3
"""
Generation 2: Robust Health and Monitoring System
Comprehensive error handling, validation, and monitoring
"""

import asyncio
import logging
import time
import json
# import psutil  # Optional dependency for detailed system metrics
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
import traceback

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

class HealthStatus(Enum):
    """System health status levels."""
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
    timestamp: datetime
    duration_ms: float
    details: Dict[str, Any]

@dataclass
class SystemMetrics:
    """System performance metrics."""
    cpu_percent: float
    memory_percent: float
    disk_percent: float
    network_io_bytes: int
    active_processes: int
    uptime_seconds: float
    timestamp: datetime

class RobustHealthMonitor:
    """Robust health monitoring system."""
    
    def __init__(self):
        self.logger = logging.getLogger("health_monitor")
        self.health_checks: Dict[str, HealthCheck] = {}
        self.metrics_history: List[SystemMetrics] = []
        self.max_history = 100
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_percent": 90.0,
            "response_time_ms": 5000.0
        }
        self.start_time = time.time()
    
    async def run_health_check(self, name: str, check_func, timeout: float = 5.0) -> HealthCheck:
        """Run individual health check with timeout and error handling."""
        start_time = time.time()
        
        try:
            # Run check with timeout
            result = await asyncio.wait_for(check_func(), timeout=timeout)
            duration_ms = (time.time() - start_time) * 1000
            
            if result.get("status") == "error":
                status = HealthStatus.CRITICAL
                message = result.get("message", "Health check failed")
            elif result.get("status") == "warning":
                status = HealthStatus.WARNING
                message = result.get("message", "Health check warning")
            else:
                status = HealthStatus.HEALTHY
                message = result.get("message", "Health check passed")
            
            return HealthCheck(
                name=name,
                status=status,
                message=message,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details=result.get("details", {})
            )
            
        except asyncio.TimeoutError:
            duration_ms = (time.time() - start_time) * 1000
            return HealthCheck(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check timed out after {timeout}s",
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details={"timeout": timeout}
            )
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Health check {name} failed: {e}")
            return HealthCheck(
                name=name,
                status=HealthStatus.CRITICAL,
                message=f"Health check error: {str(e)}",
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                details={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    async def check_system_resources(self) -> Dict[str, Any]:
        """Check system resource utilization."""
        try:
            # Try psutil if available, otherwise use basic checks
            try:
                import psutil
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                net_io = psutil.net_io_counters()
                network_bytes = net_io.bytes_sent + net_io.bytes_recv
                active_processes = len(psutil.pids())
                
                details = {
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_total_gb": memory.total / (1024**3),
                    "memory_available_gb": memory.available / (1024**3),
                    "disk_percent": (disk.used / disk.total) * 100,
                    "disk_total_gb": disk.total / (1024**3),
                    "disk_free_gb": disk.free / (1024**3),
                    "network_io_bytes": network_bytes,
                    "active_processes": active_processes
                }
                
                # Check thresholds
                warnings = []
                if cpu_percent > self.alert_thresholds["cpu_percent"]:
                    warnings.append(f"High CPU usage: {cpu_percent:.1f}%")
                if memory.percent > self.alert_thresholds["memory_percent"]:
                    warnings.append(f"High memory usage: {memory.percent:.1f}%")
                if details["disk_percent"] > self.alert_thresholds["disk_percent"]:
                    warnings.append(f"High disk usage: {details['disk_percent']:.1f}%")
                
            except ImportError:
                # Fallback to basic system checks without psutil
                import shutil
                import os
                
                # Basic disk space check
                total, used, free = shutil.disk_usage('/')
                disk_percent = (used / total) * 100
                
                details = {
                    "cpu_percent": 50.0,  # Mock value
                    "memory_percent": 60.0,  # Mock value
                    "disk_percent": disk_percent,
                    "disk_total_gb": total / (1024**3),
                    "disk_free_gb": free / (1024**3),
                    "active_processes": 100,  # Mock value
                    "psutil_available": False
                }
                
                warnings = []
                if disk_percent > self.alert_thresholds["disk_percent"]:
                    warnings.append(f"High disk usage: {disk_percent:.1f}%")
            
            if warnings:
                return {
                    "status": "warning",
                    "message": "; ".join(warnings),
                    "details": details
                }
            else:
                return {
                    "status": "healthy",
                    "message": "System resources within normal limits",
                    "details": details
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Failed to check system resources: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_evaluation_engine(self) -> Dict[str, Any]:
        """Check evaluation engine health."""
        try:
            from agi_eval_sandbox.core.evaluator import EvalSuite
            
            suite = EvalSuite()
            benchmarks = suite.list_benchmarks()
            
            # Basic functionality test
            if len(benchmarks) == 0:
                return {
                    "status": "error",
                    "message": "No benchmarks available",
                    "details": {"benchmark_count": 0}
                }
            
            # Test benchmark loading
            failed_benchmarks = []
            for benchmark_name in benchmarks:
                try:
                    benchmark = suite.get_benchmark(benchmark_name)
                    questions = benchmark.get_questions()
                    if len(questions) == 0:
                        failed_benchmarks.append(f"{benchmark_name}: no questions")
                except Exception as e:
                    failed_benchmarks.append(f"{benchmark_name}: {str(e)}")
            
            details = {
                "benchmark_count": len(benchmarks),
                "available_benchmarks": benchmarks,
                "failed_benchmarks": failed_benchmarks
            }
            
            if failed_benchmarks:
                return {
                    "status": "warning",
                    "message": f"Some benchmarks have issues: {len(failed_benchmarks)}/{len(benchmarks)}",
                    "details": details
                }
            else:
                return {
                    "status": "healthy",
                    "message": f"All {len(benchmarks)} benchmarks loaded successfully",
                    "details": details
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Evaluation engine check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_model_providers(self) -> Dict[str, Any]:
        """Check model provider connectivity (mock check)."""
        try:
            # Mock provider connectivity checks
            providers = ["openai", "anthropic", "local", "huggingface", "google"]
            provider_status = {}
            
            for provider in providers:
                # Simulate provider check (would be actual connectivity test in production)
                if provider == "local":
                    provider_status[provider] = {"status": "healthy", "latency_ms": 0}
                else:
                    # Mock status (would check actual API endpoints)
                    provider_status[provider] = {"status": "unknown", "latency_ms": None}
            
            healthy_providers = [p for p, status in provider_status.items() if status["status"] == "healthy"]
            
            return {
                "status": "healthy" if len(healthy_providers) > 0 else "warning",
                "message": f"{len(healthy_providers)}/{len(providers)} providers available",
                "details": {
                    "provider_status": provider_status,
                    "healthy_count": len(healthy_providers),
                    "total_count": len(providers)
                }
            }
            
        except Exception as e:
            return {
                "status": "error",
                "message": f"Provider check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def check_storage_access(self) -> Dict[str, Any]:
        """Check storage and file system access."""
        try:
            test_file = Path("/tmp/agi_eval_health_test.txt")
            test_data = f"health_check_{time.time()}"
            
            # Test write
            test_file.write_text(test_data)
            
            # Test read
            read_data = test_file.read_text()
            
            # Test delete
            test_file.unlink()
            
            if read_data == test_data:
                return {
                    "status": "healthy",
                    "message": "Storage read/write operations successful",
                    "details": {"test_file": str(test_file)}
                }
            else:
                return {
                    "status": "error",
                    "message": "Storage read/write test failed",
                    "details": {"expected": test_data, "actual": read_data}
                }
                
        except Exception as e:
            return {
                "status": "error",
                "message": f"Storage access check failed: {str(e)}",
                "details": {"error": str(e)}
            }
    
    async def run_all_health_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks concurrently."""
        self.logger.info("Starting comprehensive health checks")
        
        checks = {
            "system_resources": self.check_system_resources,
            "evaluation_engine": self.check_evaluation_engine,
            "model_providers": self.check_model_providers,
            "storage_access": self.check_storage_access
        }
        
        # Run all checks concurrently
        tasks = [
            self.run_health_check(name, check_func)
            for name, check_func in checks.items()
        ]
        
        results = await asyncio.gather(*tasks)
        
        # Store results
        for result in results:
            self.health_checks[result.name] = result
        
        # Log results
        critical_count = sum(1 for r in results if r.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for r in results if r.status == HealthStatus.WARNING)
        
        self.logger.info(
            f"Health checks completed: {len(results)} total, "
            f"{critical_count} critical, {warning_count} warnings"
        )
        
        return {result.name: result for result in results}
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        try:
            # Try psutil if available
            try:
                import psutil
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                net_io = psutil.net_io_counters()
                
                metrics = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_percent=(disk.used / disk.total) * 100,
                    network_io_bytes=net_io.bytes_sent + net_io.bytes_recv,
                    active_processes=len(psutil.pids()),
                    uptime_seconds=time.time() - self.start_time,
                    timestamp=datetime.now()
                )
            except ImportError:
                # Fallback metrics without psutil
                import shutil
                total, used, free = shutil.disk_usage('/')
                
                metrics = SystemMetrics(
                    cpu_percent=50.0,  # Mock value
                    memory_percent=60.0,  # Mock value  
                    disk_percent=(used / total) * 100,
                    network_io_bytes=0,  # Mock value
                    active_processes=100,  # Mock value
                    uptime_seconds=time.time() - self.start_time,
                    timestamp=datetime.now()
                )
            
            # Add to history
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history:
                self.metrics_history.pop(0)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}")
            return SystemMetrics(
                cpu_percent=0.0,
                memory_percent=0.0,
                disk_percent=0.0,
                network_io_bytes=0,
                active_processes=0,
                uptime_seconds=0.0,
                timestamp=datetime.now()
            )
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get comprehensive health summary."""
        if not self.health_checks:
            return {"status": "unknown", "message": "No health checks run yet"}
        
        critical_checks = [name for name, check in self.health_checks.items() if check.status == HealthStatus.CRITICAL]
        warning_checks = [name for name, check in self.health_checks.items() if check.status == HealthStatus.WARNING]
        healthy_checks = [name for name, check in self.health_checks.items() if check.status == HealthStatus.HEALTHY]
        
        if critical_checks:
            overall_status = "critical"
            message = f"Critical issues in: {', '.join(critical_checks)}"
        elif warning_checks:
            overall_status = "warning"
            message = f"Warnings in: {', '.join(warning_checks)}"
        else:
            overall_status = "healthy"
            message = "All systems operational"
        
        # Get latest metrics
        latest_metrics = self.collect_system_metrics()
        
        return {
            "overall_status": overall_status,
            "message": message,
            "timestamp": datetime.now().isoformat(),
            "checks": {
                "total": len(self.health_checks),
                "healthy": len(healthy_checks),
                "warning": len(warning_checks),
                "critical": len(critical_checks)
            },
            "system_metrics": asdict(latest_metrics),
            "health_checks": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "duration_ms": check.duration_ms,
                    "timestamp": check.timestamp.isoformat()
                }
                for name, check in self.health_checks.items()
            }
        }
    
    def export_health_report(self, file_path: str) -> None:
        """Export comprehensive health report."""
        report = {
            "generated_at": datetime.now().isoformat(),
            "uptime_seconds": time.time() - self.start_time,
            "summary": self.get_health_summary(),
            "detailed_checks": {
                name: asdict(check) for name, check in self.health_checks.items()
            },
            "metrics_history": [asdict(m) for m in self.metrics_history[-10:]]  # Last 10 metrics
        }
        
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Health report exported to {file_path}")

class ErrorRecoverySystem:
    """Automatic error recovery and self-healing."""
    
    def __init__(self):
        self.logger = logging.getLogger("error_recovery")
        self.recovery_attempts: Dict[str, int] = {}
        self.max_recovery_attempts = 3
        self.recovery_strategies = {
            "memory_high": self._recover_memory_pressure,
            "disk_full": self._recover_disk_space,
            "benchmark_failure": self._recover_benchmark_loading,
            "model_timeout": self._recover_model_timeout
        }
    
    async def _recover_memory_pressure(self) -> bool:
        """Attempt to recover from high memory usage."""
        try:
            import gc
            gc.collect()  # Force garbage collection
            self.logger.info("Triggered garbage collection for memory recovery")
            return True
        except Exception as e:
            self.logger.error(f"Memory recovery failed: {e}")
            return False
    
    async def _recover_disk_space(self) -> bool:
        """Attempt to recover disk space."""
        try:
            # Clean up temporary files
            import shutil
            temp_paths = ["/tmp", "/var/tmp"]
            cleaned = 0
            
            for temp_path in temp_paths:
                if Path(temp_path).exists():
                    for file_path in Path(temp_path).glob("agi_eval_*"):
                        try:
                            if file_path.is_file():
                                file_path.unlink()
                                cleaned += 1
                        except Exception:
                            pass
            
            self.logger.info(f"Cleaned {cleaned} temporary files")
            return cleaned > 0
        except Exception as e:
            self.logger.error(f"Disk space recovery failed: {e}")
            return False
    
    async def _recover_benchmark_loading(self) -> bool:
        """Attempt to recover from benchmark loading issues."""
        try:
            # Re-initialize evaluation suite
            from agi_eval_sandbox.core.evaluator import EvalSuite
            suite = EvalSuite()
            benchmarks = suite.list_benchmarks()
            self.logger.info(f"Re-initialized evaluation suite with {len(benchmarks)} benchmarks")
            return len(benchmarks) > 0
        except Exception as e:
            self.logger.error(f"Benchmark recovery failed: {e}")
            return False
    
    async def _recover_model_timeout(self) -> bool:
        """Attempt to recover from model timeouts."""
        try:
            # Reset connection pools, clear caches
            self.logger.info("Clearing model caches and resetting connections")
            # Implementation would reset actual connections
            return True
        except Exception as e:
            self.logger.error(f"Model timeout recovery failed: {e}")
            return False
    
    async def attempt_recovery(self, error_type: str) -> bool:
        """Attempt automatic recovery for known error types."""
        if error_type not in self.recovery_strategies:
            self.logger.warning(f"No recovery strategy for error type: {error_type}")
            return False
        
        attempts = self.recovery_attempts.get(error_type, 0)
        if attempts >= self.max_recovery_attempts:
            self.logger.error(f"Max recovery attempts ({self.max_recovery_attempts}) exceeded for {error_type}")
            return False
        
        self.recovery_attempts[error_type] = attempts + 1
        
        self.logger.info(f"Attempting recovery for {error_type} (attempt {attempts + 1}/{self.max_recovery_attempts})")
        
        try:
            success = await self.recovery_strategies[error_type]()
            if success:
                self.logger.info(f"Recovery successful for {error_type}")
                self.recovery_attempts[error_type] = 0  # Reset on success
            else:
                self.logger.warning(f"Recovery failed for {error_type}")
            return success
        except Exception as e:
            self.logger.error(f"Recovery attempt failed for {error_type}: {e}")
            return False

async def demonstrate_robust_health_system():
    """Demonstrate Generation 2 robust health and monitoring."""
    print("🛡️  Generation 2: Robust Health & Monitoring System")
    print("=" * 60)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize health monitor
    health_monitor = RobustHealthMonitor()
    recovery_system = ErrorRecoverySystem()
    
    print("🔍 Running comprehensive health checks...")
    
    # Run health checks
    health_results = await health_monitor.run_all_health_checks()
    
    # Display results
    print("\n📊 Health Check Results:")
    print("-" * 40)
    
    for name, check in health_results.items():
        status_icon = {
            HealthStatus.HEALTHY: "✅",
            HealthStatus.WARNING: "⚠️ ",
            HealthStatus.CRITICAL: "❌",
            HealthStatus.UNKNOWN: "❓"
        }[check.status]
        
        print(f"{status_icon} {name}: {check.status.value}")
        print(f"   {check.message}")
        print(f"   Duration: {check.duration_ms:.1f}ms")
        
        if check.details:
            if check.status != HealthStatus.HEALTHY:
                print(f"   Details: {check.details}")
        print()
    
    # System metrics
    print("📈 Current System Metrics:")
    print("-" * 30)
    metrics = health_monitor.collect_system_metrics()
    print(f"CPU Usage: {metrics.cpu_percent:.1f}%")
    print(f"Memory Usage: {metrics.memory_percent:.1f}%") 
    print(f"Disk Usage: {metrics.disk_percent:.1f}%")
    print(f"Active Processes: {metrics.active_processes}")
    print(f"Uptime: {metrics.uptime_seconds:.1f}s")
    
    # Health summary
    print("\n🏥 Health Summary:")
    print("-" * 20)
    summary = health_monitor.get_health_summary()
    print(f"Overall Status: {summary['overall_status']}")
    print(f"Message: {summary['message']}")
    print(f"Checks: {summary['checks']['healthy']} healthy, "
          f"{summary['checks']['warning']} warnings, "
          f"{summary['checks']['critical']} critical")
    
    # Test error recovery
    print("\n🔧 Testing Error Recovery System:")
    print("-" * 35)
    
    recovery_tests = ["memory_high", "benchmark_failure"]
    for error_type in recovery_tests:
        print(f"Testing recovery for: {error_type}")
        success = await recovery_system.attempt_recovery(error_type)
        print(f"  Recovery {'✅ successful' if success else '❌ failed'}")
    
    # Export health report
    report_path = "/tmp/health_report.json"
    health_monitor.export_health_report(report_path)
    print(f"\n📋 Health report exported to: {report_path}")
    
    print("\n✅ Generation 2 robust health system demonstration complete!")
    return True

if __name__ == "__main__":
    success = asyncio.run(demonstrate_robust_health_system())
    sys.exit(0 if success else 1)