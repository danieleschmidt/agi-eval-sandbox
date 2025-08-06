"""Structured logging configuration."""

import logging
import logging.config
import logging.handlers
from typing import Dict, Any, Optional, List
import json
import traceback
import threading
import queue
import time
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from contextlib import contextmanager
import os
import sys
import socket
from pathlib import Path


@dataclass
class LogMetrics:
    """Metrics for log monitoring."""
    total_logs: int = 0
    errors: int = 0
    warnings: int = 0
    performance_events: int = 0
    security_events: int = 0
    last_error_time: Optional[datetime] = None
    last_warning_time: Optional[datetime] = None
    start_time: datetime = field(default_factory=datetime.now)


class EnhancedStructuredFormatter(logging.Formatter):
    """Enhanced formatter for structured JSON logging with correlation IDs and context."""
    
    def __init__(self, include_trace_id: bool = True, include_system_info: bool = True):
        super().__init__()
        self.include_trace_id = include_trace_id
        self.include_system_info = include_system_info
        self.hostname = socket.gethostname()
        self.pid = os.getpid()
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as enhanced structured JSON."""
        # Base log entry with improved timestamp format
        log_entry = {
            "@timestamp": datetime.fromtimestamp(record.created).isoformat() + 'Z',
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "source": {
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno,
                "file": record.pathname
            },
            "thread": {
                "id": record.thread,
                "name": record.threadName
            }
        }
        
        # Add system information if enabled
        if self.include_system_info:
            log_entry["system"] = {
                "hostname": self.hostname,
                "pid": self.pid,
                "process_name": record.processName
            }
        
        # Add trace/correlation ID if available
        if self.include_trace_id and hasattr(record, 'trace_id'):
            log_entry["trace_id"] = record.trace_id
        if hasattr(record, 'correlation_id'):
            log_entry["correlation_id"] = record.correlation_id
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields from record
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'getMessage', 'exc_info', 'exc_text',
                          'stack_info']:
                log_entry[key] = value
        
        return json.dumps(log_entry, default=str)


class ContextFilter(logging.Filter):
    """Add contextual information to log records."""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add context to log record."""
        # Add process and thread info
        record.process_name = record.processName
        record.thread_name = record.threadName
        
        # Add environment info
        record.environment = os.getenv('ENVIRONMENT', 'development')
        
        return True


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    structured: bool = True,
    include_console: bool = True
) -> None:
    """Setup application logging configuration."""
    
    config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "structured": {
                "()": StructuredFormatter,
            },
            "simple": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            }
        },
        "filters": {
            "context": {
                "()": ContextFilter,
            }
        },
        "handlers": {},
        "loggers": {
            "agi_eval_sandbox": {
                "level": level,
                "handlers": [],
                "propagate": False
            },
            "uvicorn": {
                "level": "INFO",
                "handlers": [],
                "propagate": False
            },
            "fastapi": {
                "level": "INFO", 
                "handlers": [],
                "propagate": False
            }
        },
        "root": {
            "level": level,
            "handlers": []
        }
    }
    
    handler_names = []
    
    # Console handler
    if include_console:
        config["handlers"]["console"] = {
            "class": "logging.StreamHandler",
            "level": level,
            "formatter": "structured" if structured else "simple",
            "filters": ["context"],
            "stream": "ext://sys.stdout"
        }
        handler_names.append("console")
    
    # File handler
    if log_file:
        config["handlers"]["file"] = {
            "class": "logging.handlers.RotatingFileHandler",
            "level": level,
            "formatter": "structured",
            "filters": ["context"],
            "filename": log_file,
            "maxBytes": 10485760,  # 10MB
            "backupCount": 5,
            "encoding": "utf8"
        }
        handler_names.append("file")
    
    # Assign handlers to loggers
    for logger_name in ["agi_eval_sandbox", "uvicorn", "fastapi"]:
        config["loggers"][logger_name]["handlers"] = handler_names
    
    config["root"]["handlers"] = handler_names
    
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    """Get a configured logger instance."""
    return logging.getLogger(f"agi_eval_sandbox.{name}")


class SecurityLogger:
    """Specialized logger for security events."""
    
    def __init__(self):
        self.logger = get_logger("security")
    
    def log_authentication_failure(self, user_id: Optional[str], ip_address: str, reason: str):
        """Log authentication failure."""
        self.logger.warning(
            "Authentication failure",
            extra={
                "event_type": "auth_failure",
                "user_id": user_id,
                "ip_address": ip_address,
                "reason": reason
            }
        )
    
    def log_authorization_failure(self, user_id: str, resource: str, action: str):
        """Log authorization failure."""
        self.logger.warning(
            "Authorization failure",
            extra={
                "event_type": "authz_failure",
                "user_id": user_id,
                "resource": resource,
                "action": action
            }
        )
    
    def log_suspicious_activity(self, activity: str, details: Dict[str, Any]):
        """Log suspicious activity."""
        self.logger.warning(
            f"Suspicious activity: {activity}",
            extra={
                "event_type": "suspicious_activity",
                "activity": activity,
                **details
            }
        )
    
    def log_security_violation(self, violation: str, details: Dict[str, Any]):
        """Log security violation."""
        self.logger.error(
            f"Security violation: {violation}",
            extra={
                "event_type": "security_violation",
                "violation": violation,
                **details
            }
        )


class PerformanceLogger:
    """Specialized logger for performance metrics."""
    
    def __init__(self):
        self.logger = get_logger("performance")
    
    def log_evaluation_performance(
        self, 
        model_name: str, 
        benchmark: str, 
        duration_seconds: float,
        questions_count: int,
        success: bool
    ):
        """Log evaluation performance metrics."""
        self.logger.info(
            "Evaluation performance",
            extra={
                "event_type": "evaluation_performance",
                "model_name": model_name,
                "benchmark": benchmark,
                "duration_seconds": duration_seconds,
                "questions_count": questions_count,
                "questions_per_second": questions_count / duration_seconds if duration_seconds > 0 else 0,
                "success": success
            }
        )
    
    def log_api_performance(
        self, 
        endpoint: str, 
        method: str, 
        duration_ms: float,
        status_code: int,
        response_size_bytes: int
    ):
        """Log API performance metrics."""
        self.logger.info(
            "API performance",
            extra={
                "event_type": "api_performance",
                "endpoint": endpoint,
                "method": method,
                "duration_ms": duration_ms,
                "status_code": status_code,
                "response_size_bytes": response_size_bytes
            }
        )
    
    def log_resource_usage(
        self,
        memory_mb: float,
        cpu_percent: float,
        disk_usage_mb: float
    ):
        """Log resource usage metrics."""
        self.logger.info(
            "Resource usage",
            extra={
                "event_type": "resource_usage",
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "disk_usage_mb": disk_usage_mb
            }
        )


class AuditLogger:
    """Specialized logger for audit trails and compliance."""
    
    def __init__(self):
        self.logger = get_logger("audit")
        self.metrics = LogMetrics()
    
    def log_user_action(self, user_id: str, action: str, resource: str, result: str, details: Optional[Dict[str, Any]] = None):
        """Log user actions for audit trail."""
        self.logger.info(
            f"User action: {action}",
            extra={
                "event_type": "user_action",
                "user_id": user_id,
                "action": action,
                "resource": resource,
                "result": result,
                "details": details or {},
                "compliance": {
                    "audit_required": True,
                    "retention_period": "7_years"
                }
            }
        )
        self.metrics.total_logs += 1
    
    def log_data_access(self, user_id: str, data_type: str, operation: str, records_affected: int):
        """Log data access for privacy compliance."""
        self.logger.info(
            f"Data access: {operation} on {data_type}",
            extra={
                "event_type": "data_access",
                "user_id": user_id,
                "data_type": data_type,
                "operation": operation,
                "records_affected": records_affected,
                "compliance": {
                    "gdpr_relevant": True,
                    "audit_required": True
                }
            }
        )
        self.metrics.total_logs += 1
    
    def log_system_change(self, user_id: str, change_type: str, component: str, before: Any, after: Any):
        """Log system configuration changes."""
        self.logger.info(
            f"System change: {change_type} in {component}",
            extra={
                "event_type": "system_change",
                "user_id": user_id,
                "change_type": change_type,
                "component": component,
                "before": str(before)[:1000],  # Truncate for log size
                "after": str(after)[:1000],
                "compliance": {
                    "audit_required": True,
                    "change_control": True
                }
            }
        )
        self.metrics.total_logs += 1


class CorrelationIdFilter(logging.Filter):
    """Filter to add correlation IDs to log records."""
    
    def __init__(self):
        super().__init__()
        self._local = threading.local()
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current thread."""
        self._local.correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get correlation ID for current thread."""
        return getattr(self._local, 'correlation_id', None)
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add correlation ID to log record."""
        correlation_id = self.get_correlation_id()
        if correlation_id:
            record.correlation_id = correlation_id
        return True


class MetricsCollectingHandler(logging.Handler):
    """Handler that collects logging metrics."""
    
    def __init__(self):
        super().__init__()
        self.metrics = LogMetrics()
        self._log_counts = {}
        self._lock = threading.Lock()
    
    def emit(self, record: logging.LogRecord):
        """Collect metrics from log record."""
        with self._lock:
            self.metrics.total_logs += 1
            
            if record.levelno >= logging.ERROR:
                self.metrics.errors += 1
                self.metrics.last_error_time = datetime.now()
            elif record.levelno >= logging.WARNING:
                self.metrics.warnings += 1
                self.metrics.last_warning_time = datetime.now()
            
            # Count by logger
            logger_name = record.name
            if logger_name not in self._log_counts:
                self._log_counts[logger_name] = 0
            self._log_counts[logger_name] += 1
            
            # Count special event types
            if hasattr(record, 'event_type'):
                if record.event_type == 'performance':
                    self.metrics.performance_events += 1
                elif record.event_type in ['security_violation', 'suspicious_activity', 'auth_failure', 'authz_failure']:
                    self.metrics.security_events += 1
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get collected metrics."""
        with self._lock:
            uptime = datetime.now() - self.metrics.start_time
            return {
                "total_logs": self.metrics.total_logs,
                "errors": self.metrics.errors,
                "warnings": self.metrics.warnings,
                "performance_events": self.metrics.performance_events,
                "security_events": self.metrics.security_events,
                "last_error_time": self.metrics.last_error_time.isoformat() if self.metrics.last_error_time else None,
                "last_warning_time": self.metrics.last_warning_time.isoformat() if self.metrics.last_warning_time else None,
                "uptime_seconds": uptime.total_seconds(),
                "log_counts_by_logger": self._log_counts.copy(),
                "logs_per_second": self.metrics.total_logs / uptime.total_seconds() if uptime.total_seconds() > 0 else 0
            }


@contextmanager
def correlation_context(correlation_id: str):
    """Context manager for setting correlation ID."""
    correlation_filter = correlation_id_filter  # Global filter instance
    old_id = correlation_filter.get_correlation_id()
    correlation_filter.set_correlation_id(correlation_id)
    try:
        yield correlation_id
    finally:
        if old_id:
            correlation_filter.set_correlation_id(old_id)
        else:
            correlation_filter.set_correlation_id(None)


# Global instances
correlation_id_filter = CorrelationIdFilter()
metrics_handler = MetricsCollectingHandler()

# Initialize loggers
security_logger = SecurityLogger()
performance_logger = PerformanceLogger()
audit_logger = AuditLogger()