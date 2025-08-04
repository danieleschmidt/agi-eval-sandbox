"""Structured logging configuration."""

import logging
import logging.config
from typing import Dict, Any, Optional
import json
import traceback
from datetime import datetime
import os


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
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


# Initialize security and performance loggers
security_logger = SecurityLogger()
performance_logger = PerformanceLogger()