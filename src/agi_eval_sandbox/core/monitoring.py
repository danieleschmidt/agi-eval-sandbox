"""Enhanced monitoring and metrics collection for AGI Evaluation Sandbox."""

import time
import asyncio
import psutil
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

from .logging_config import get_logger, performance_logger
from .exceptions import ResourceError

logger = get_logger("monitoring")


@dataclass
class SystemMetrics:
    """System resource metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_mb: float = 0.0
    memory_available_mb: float = 0.0
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0
    active_connections: int = 0


@dataclass
class EvaluationMetrics:
    """Evaluation performance metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    total_evaluations: int = 0
    active_evaluations: int = 0
    avg_response_time_ms: float = 0.0
    throughput_qps: float = 0.0
    success_rate: float = 100.0
    error_rate: float = 0.0
    cache_hit_rate: float = 0.0


class MetricsCollector:
    """Collects and aggregates system and evaluation metrics."""
    
    def __init__(self, collection_interval: int = 60):
        self.collection_interval = collection_interval
        self.system_metrics_history: List[SystemMetrics] = []
        self.evaluation_metrics_history: List[EvaluationMetrics] = []
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Evaluation counters
        self.total_evaluations = 0
        self.successful_evaluations = 0
        self.failed_evaluations = 0
        self.total_response_time = 0.0
    
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
    
    async def _collection_loop(self):
        """Main collection loop."""
        while self._running:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                self.system_metrics_history.append(system_metrics)
                
                # Collect evaluation metrics
                eval_metrics = self._collect_evaluation_metrics()
                self.evaluation_metrics_history.append(eval_metrics)
                
                # Trim history to last 24 hours
                self._trim_history()
                
                # Log metrics
                performance_logger.log_system_metrics(
                    cpu_percent=system_metrics.cpu_percent,
                    memory_percent=system_metrics.memory_percent,
                    disk_usage_percent=system_metrics.disk_usage_percent
                )
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                await asyncio.sleep(self.collection_interval)
    
    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)
            
            # Disk usage
            disk = psutil.disk_usage('/')
            disk_usage_percent = (disk.used / disk.total) * 100
            
            # Network usage
            network = psutil.net_io_counters()
            
            # Active connections
            try:
                connections = len(psutil.net_connections())
            except (psutil.AccessDenied, OSError):
                connections = 0
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                disk_usage_percent=disk_usage_percent,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                active_connections=connections
            )
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return SystemMetrics()
    
    def _collect_evaluation_metrics(self) -> EvaluationMetrics:
        """Collect evaluation performance metrics."""
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
            
            avg_response_time_ms = (
                (self.total_response_time / self.total_evaluations * 1000)
                if self.total_evaluations > 0 else 0.0
            )
            
            # Calculate throughput (evaluations per second over last minute)
            throughput_qps = self.total_evaluations / max(self.collection_interval, 1)
            
            return EvaluationMetrics(
                total_evaluations=self.total_evaluations,
                active_evaluations=0,  # Would be set by evaluator
                avg_response_time_ms=avg_response_time_ms,
                throughput_qps=throughput_qps,
                success_rate=success_rate,
                error_rate=error_rate,
                cache_hit_rate=0.0  # Would be set by cache manager
            )
            
        except Exception as e:
            logger.error(f"Error collecting evaluation metrics: {e}")
            return EvaluationMetrics()
    
    def _trim_history(self):
        """Trim metrics history to last 24 hours."""
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        self.system_metrics_history = [
            m for m in self.system_metrics_history
            if m.timestamp > cutoff_time
        ]
        
        self.evaluation_metrics_history = [
            m for m in self.evaluation_metrics_history
            if m.timestamp > cutoff_time
        ]
    
    def record_evaluation_start(self):
        """Record the start of an evaluation."""
        self.total_evaluations += 1
    
    def record_evaluation_success(self, duration_seconds: float):
        """Record a successful evaluation."""
        self.successful_evaluations += 1
        self.total_response_time += duration_seconds
    
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


# Global metrics collector instance
metrics_collector = MetricsCollector()