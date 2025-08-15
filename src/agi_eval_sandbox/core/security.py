"""Enhanced security utilities and validation for AGI Evaluation Sandbox."""

import re
import hashlib
import secrets
import hmac
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import json

from .logging_config import get_logger, security_logger
from .exceptions import SecurityError, ValidationError

logger = get_logger("security")


@dataclass
class SecurityEvent:
    """Represents a security-related event."""
    timestamp: datetime
    event_type: str  # "authentication", "authorization", "input_validation", "rate_limit", etc.
    severity: str    # "low", "medium", "high", "critical"
    source_ip: Optional[str] = None
    user_id: Optional[str] = None
    details: Dict[str, Any] = None


class InputSanitizer:
    """Comprehensive input sanitization and validation."""
    
    # Patterns for detecting potentially malicious content
    SQL_INJECTION_PATTERNS = [
        r"(\bUNION\b.*\bSELECT\b)",
        r"(\bSELECT\b.*\bFROM\b)",
        r"(\bINSERT\b.*\bINTO\b)",
        r"(\bUPDATE\b.*\bSET\b)",
        r"(\bDELETE\b.*\bFROM\b)",
        r"(\bDROP\b.*\bTABLE\b)",
        r"(\';|\-\-|\/\*|\*\/)",
    ]
    
    XSS_PATTERNS = [
        r"<script[^>]*>.*?</script>",
        r"javascript:",
        r"on\w+\s*=",
        r"<iframe[^>]*>",
        r"<object[^>]*>",
        r"<embed[^>]*>",
    ]
    
    COMMAND_INJECTION_PATTERNS = [
        r"[;&|`$]",
        r"\$\([^)]*\)",
        r"`[^`]*`",
        r"\$\{[^}]*\}",
    ]
    
    def __init__(self):
        self.sql_patterns = [re.compile(p, re.IGNORECASE) for p in self.SQL_INJECTION_PATTERNS]
        self.xss_patterns = [re.compile(p, re.IGNORECASE) for p in self.XSS_PATTERNS]
        self.cmd_patterns = [re.compile(p) for p in self.COMMAND_INJECTION_PATTERNS]
    
    def sanitize_input(self, input_text: str, max_length: int = 10000) -> str:
        """Sanitize user input with multiple security checks."""
        if not isinstance(input_text, str):
            raise ValidationError("Input must be a string")
        
        # Length check
        if len(input_text) > max_length:
            raise ValidationError(f"Input too long: {len(input_text)} > {max_length}")
        
        # Check for malicious patterns
        self._check_sql_injection(input_text)
        self._check_xss(input_text)
        self._check_command_injection(input_text)
        
        # Remove null bytes and control characters
        sanitized = input_text.replace('\x00', '').replace('\r', '').replace('\n', ' ')
        
        # Normalize whitespace
        sanitized = ' '.join(sanitized.split())
        
        return sanitized
    
    def _check_sql_injection(self, text: str):
        """Check for SQL injection patterns."""
        for pattern in self.sql_patterns:
            if pattern.search(text):
                security_logger.log_security_violation(
                    "sql_injection_attempt",
                    {"pattern": pattern.pattern, "text_snippet": text[:100]}
                )
                raise SecurityError("Potential SQL injection detected")
    
    def _check_xss(self, text: str):
        """Check for XSS patterns."""
        for pattern in self.xss_patterns:
            if pattern.search(text):
                security_logger.log_security_violation(
                    "xss_attempt", 
                    {"pattern": pattern.pattern, "text_snippet": text[:100]}
                )
                raise SecurityError("Potential XSS attack detected")
    
    def _check_command_injection(self, text: str):
        """Check for command injection patterns."""
        for pattern in self.cmd_patterns:
            if pattern.search(text):
                security_logger.log_security_violation(
                    "command_injection_attempt",
                    {"pattern": pattern.pattern, "text_snippet": text[:100]}
                )
                raise SecurityError("Potential command injection detected")
    
    def validate_api_key(self, api_key: str) -> bool:
        """Validate API key format and structure."""
        if not api_key:
            return False
        
        # Basic format checks
        if len(api_key) < 20 or len(api_key) > 200:
            return False
        
        # Check for valid characters (alphanumeric, hyphens, underscores)
        if not re.match(r'^[a-zA-Z0-9\-_]+$', api_key):
            return False
        
        return True
    
    def validate_model_name(self, model_name: str) -> bool:
        """Validate model name format."""
        if not model_name or len(model_name) > 100:
            return False
        
        # Allow alphanumeric, hyphens, underscores, and dots
        if not re.match(r'^[a-zA-Z0-9\-_.]+$', model_name):
            return False
        
        return True


class RateLimiter:
    """Advanced rate limiting with multiple strategies."""
    
    def __init__(self):
        self.request_history: Dict[str, List[float]] = {}
        self.blocked_ips: Dict[str, float] = {}  # IP -> block_until_timestamp
        self.security_events: List[SecurityEvent] = []
    
    def check_rate_limit(
        self, 
        identifier: str, 
        max_requests: int = 100, 
        window_seconds: int = 3600,
        block_duration_seconds: int = 300
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if request is within rate limits.
        
        Returns:
            (is_allowed, reason_if_blocked)
        """
        current_time = time.time()
        
        # Check if IP is currently blocked
        if identifier in self.blocked_ips:
            if current_time < self.blocked_ips[identifier]:
                remaining_time = int(self.blocked_ips[identifier] - current_time)
                return False, f"IP blocked for {remaining_time} more seconds"
            else:
                del self.blocked_ips[identifier]
        
        # Initialize request history for new identifiers
        if identifier not in self.request_history:
            self.request_history[identifier] = []
        
        # Clean old requests outside the window
        window_start = current_time - window_seconds
        self.request_history[identifier] = [
            req_time for req_time in self.request_history[identifier]
            if req_time > window_start
        ]
        
        # Check if within rate limit
        if len(self.request_history[identifier]) >= max_requests:
            # Block the IP
            self.blocked_ips[identifier] = current_time + block_duration_seconds
            
            # Log security event
            event = SecurityEvent(
                timestamp=datetime.now(),
                event_type="rate_limit_exceeded",
                severity="medium",
                source_ip=identifier,
                details={
                    "requests_in_window": len(self.request_history[identifier]),
                    "max_requests": max_requests,
                    "window_seconds": window_seconds
                }
            )
            self.security_events.append(event)
            security_logger.log_security_violation(
                "rate_limit_exceeded", 
                event.details
            )
            
            return False, f"Rate limit exceeded: {max_requests} requests per {window_seconds} seconds"
        
        # Record this request
        self.request_history[identifier].append(current_time)
        return True, None
    
    def get_rate_limit_stats(self, identifier: str) -> Dict[str, Any]:
        """Get rate limit statistics for an identifier."""
        current_time = time.time()
        
        if identifier not in self.request_history:
            return {
                "requests_in_current_window": 0,
                "is_blocked": False,
                "block_remaining_seconds": 0
            }
        
        # Count recent requests
        window_start = current_time - 3600  # Default 1 hour window
        recent_requests = len([
            req_time for req_time in self.request_history[identifier]
            if req_time > window_start
        ])
        
        # Check block status
        is_blocked = identifier in self.blocked_ips and current_time < self.blocked_ips[identifier]
        block_remaining = 0
        if is_blocked:
            block_remaining = int(self.blocked_ips[identifier] - current_time)
        
        return {
            "requests_in_current_window": recent_requests,
            "is_blocked": is_blocked,
            "block_remaining_seconds": block_remaining
        }


class SecurityAuditor:
    """Security auditing and compliance checking."""
    
    def __init__(self):
        self.security_events: List[SecurityEvent] = []
        self.input_sanitizer = InputSanitizer()
        self.rate_limiter = RateLimiter()
    
    def audit_request(
        self, 
        request_data: Dict[str, Any], 
        source_ip: Optional[str] = None,
        user_id: Optional[str] = None
    ) -> List[SecurityEvent]:
        """Perform comprehensive security audit of a request."""
        events = []
        
        try:
            # Rate limiting check
            if source_ip:
                is_allowed, reason = self.rate_limiter.check_rate_limit(source_ip)
                if not is_allowed:
                    events.append(SecurityEvent(
                        timestamp=datetime.now(),
                        event_type="rate_limit_violation",
                        severity="medium",
                        source_ip=source_ip,
                        user_id=user_id,
                        details={"reason": reason}
                    ))
            
            # Input validation
            for key, value in request_data.items():
                if isinstance(value, str):
                    try:
                        self.input_sanitizer.sanitize_input(value)
                    except (SecurityError, ValidationError) as e:
                        events.append(SecurityEvent(
                            timestamp=datetime.now(),
                            event_type="malicious_input_detected",
                            severity="high",
                            source_ip=source_ip,
                            user_id=user_id,
                            details={
                                "field": key,
                                "error": str(e),
                                "value_snippet": str(value)[:100]
                            }
                        ))
            
            # API key validation
            if 'api_key' in request_data:
                api_key = request_data['api_key']
                if not self.input_sanitizer.validate_api_key(api_key):
                    events.append(SecurityEvent(
                        timestamp=datetime.now(),
                        event_type="invalid_api_key",
                        severity="medium",
                        source_ip=source_ip,
                        user_id=user_id,
                        details={"api_key_length": len(api_key) if api_key else 0}
                    ))
            
            # Model name validation
            if 'model' in request_data and isinstance(request_data['model'], dict):
                model_name = request_data['model'].get('name')
                if model_name and not self.input_sanitizer.validate_model_name(model_name):
                    events.append(SecurityEvent(
                        timestamp=datetime.now(),
                        event_type="invalid_model_name",
                        severity="low",
                        source_ip=source_ip,
                        user_id=user_id,
                        details={"model_name": model_name}
                    ))
            
        except Exception as e:
            logger.error(f"Error during security audit: {e}")
            events.append(SecurityEvent(
                timestamp=datetime.now(),
                event_type="audit_error",
                severity="high",
                source_ip=source_ip,
                user_id=user_id,
                details={"error": str(e)}
            ))
        
        # Store events
        self.security_events.extend(events)
        
        # Log high/critical events
        for event in events:
            if event.severity in ["high", "critical"]:
                security_logger.log_security_violation(
                    event.event_type,
                    event.details or {}
                )
        
        return events
    
    def get_security_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get security events summary for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            event for event in self.security_events
            if event.timestamp > cutoff_time
        ]
        
        # Count by severity
        severity_counts = {}
        event_type_counts = {}
        
        for event in recent_events:
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            event_type_counts[event.event_type] = event_type_counts.get(event.event_type, 0) + 1
        
        return {
            "time_period_hours": hours,
            "total_events": len(recent_events),
            "severity_breakdown": severity_counts,
            "event_type_breakdown": event_type_counts,
            "unique_source_ips": len(set(
                event.source_ip for event in recent_events 
                if event.source_ip
            )),
            "blocked_ips_count": len(self.rate_limiter.blocked_ips)
        }
    
    def export_security_log(self, file_path: str, hours: int = 24):
        """Export security events to JSON file."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_events = [
            {
                "timestamp": event.timestamp.isoformat(),
                "event_type": event.event_type,
                "severity": event.severity,
                "source_ip": event.source_ip,
                "user_id": event.user_id,
                "details": event.details
            }
            for event in self.security_events
            if event.timestamp > cutoff_time
        ]
        
        data = {
            "export_timestamp": datetime.now().isoformat(),
            "time_period_hours": hours,
            "total_events": len(recent_events),
            "events": recent_events
        }
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Exported {len(recent_events)} security events to {file_path}")


# Global security auditor instance
security_auditor = SecurityAuditor()