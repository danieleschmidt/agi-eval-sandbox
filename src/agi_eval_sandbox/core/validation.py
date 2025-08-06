"""Input validation and sanitization utilities."""

import re
import json
import hashlib
import secrets
import ipaddress
from urllib.parse import urlparse
from typing import Any, Dict, List, Optional, Union, Set
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field

from .exceptions import ValidationError, SecurityError, ResourceError
from .logging_config import get_logger, security_logger

logger = get_logger("validation")


@dataclass
class ValidationConfig:
    """Configuration for validation rules."""
    max_string_length: int = 10000
    max_json_size: int = 100000
    max_file_size: int = 50 * 1024 * 1024  # 50MB
    allowed_file_extensions: Set[str] = field(default_factory=lambda: {
        '.json', '.txt', '.csv', '.yaml', '.yml', '.py'
    })
    rate_limit_window: int = 60  # seconds
    max_requests_per_window: int = 100


class InputValidator:
    """Comprehensive input validation with security features."""
    
    # Security patterns to detect potential threats
    DANGEROUS_PATTERNS = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',  # JavaScript protocol
        r'vbscript:',   # VBScript protocol
        r'on\w+\s*=',   # Event handlers
        r'eval\s*\(',   # eval() calls
        r'exec\s*\(',   # exec() calls
        r'import\s+os', # OS imports
        r'__import__',  # Dynamic imports
        r'\.\./',       # Path traversal
        r'\\.\\.\\',    # Windows path traversal
        # Additional enhanced patterns
        r'data:text/html',  # Data URI HTML
        r'Function\s*\(',  # Function constructor
        r'setTimeout\s*\(',  # setTimeout calls
        r'setInterval\s*\(',  # setInterval calls
        r'import\s+subprocess', # subprocess imports  
        r'getattr\s*\(',  # getattr calls
        r'setattr\s*\(',  # setattr calls
        r'compile\s*\(',  # compile calls
        r'open\s*\(',  # file operations
        r'~/',         # Home directory access
        r'\/etc\/',     # System directory access
        r'\/proc\/',    # Process directory access
        r'\/sys\/',     # System directory access
        r'(union|select|insert|update|delete|drop|create|alter)\s+',  # SQL injection
        r'(or|and)\s+\d+\s*=\s*\d+',  # SQL injection
        r';\s*(rm|del|format|shutdown|reboot)',  # Command injection
        r'\|\s*(rm|del|format|shutdown|reboot)',  # Command injection
        r'&&\s*(rm|del|format|shutdown|reboot)',  # Command injection
        r'`[^`]*`',  # Backtick command execution
        r'\$\([^)]*\)',  # Command substitution
    ]
    
    # Suspicious keywords that might indicate malicious intent
    SUSPICIOUS_KEYWORDS = [
        'password', 'secret', 'token', 'key', 'credential',
        'admin', 'root', 'sudo', 'administrator',
        'backdoor', 'exploit', 'payload', 'shell',
        'malware', 'virus', 'trojan', 'ransomware'
    ]
    
    def __init__(self, config: Optional[ValidationConfig] = None):
        self.config = config or ValidationConfig()
        self._request_counts: Dict[str, List[datetime]] = {}
        self._blocked_ips: Set[str] = set()
        self._suspicious_patterns_cache: Dict[str, bool] = {}
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limits."""
        now = datetime.now()
        
        if client_id not in self._request_counts:
            self._request_counts[client_id] = []
        
        # Clean old requests outside the window
        self._request_counts[client_id] = [
            req_time for req_time in self._request_counts[client_id]
            if now - req_time < timedelta(seconds=self.config.rate_limit_window)
        ]
        
        # Check if over limit
        if len(self._request_counts[client_id]) >= self.config.max_requests_per_window:
            security_logger.log_suspicious_activity(
                "Rate limit exceeded",
                {"client_id": client_id, "requests": len(self._request_counts[client_id])}
            )
            return False
        
        # Record this request
        self._request_counts[client_id].append(now)
        return True
    
    def is_ip_blocked(self, ip_address: str) -> bool:
        """Check if IP address is blocked."""
        return ip_address in self._blocked_ips
    
    def block_ip(self, ip_address: str, reason: str) -> None:
        """Block an IP address."""
        self._blocked_ips.add(ip_address)
        security_logger.log_security_violation(
            f"IP address blocked: {reason}",
            {"ip_address": ip_address, "reason": reason}
        )
    
    def validate_ip_address(self, ip_str: str) -> str:
        """Validate IP address format."""
        try:
            ip = ipaddress.ip_address(ip_str)
            # Block private/internal addresses in certain contexts
            if ip.is_private or ip.is_loopback or ip.is_reserved:
                logger.warning(f"Internal IP address used: {ip_str}")
            return str(ip)
        except ValueError:
            raise ValidationError(f"Invalid IP address: {ip_str}")
    
    def validate_url(self, url: str, allow_local: bool = False) -> str:
        """Validate URL format and security."""
        url = self.validate_string(url, "url", max_length=2000)
        
        try:
            parsed = urlparse(url)
            
            # Check scheme
            if parsed.scheme not in ['http', 'https']:
                raise ValidationError(f"Unsupported URL scheme: {parsed.scheme}")
            
            # Check for suspicious patterns in URL
            if any(re.search(pattern, url, re.IGNORECASE) for pattern in self.DANGEROUS_PATTERNS[:10]):
                raise SecurityError(f"Potentially dangerous URL: {url}")
            
            # Validate hostname if present
            if parsed.hostname:
                if not allow_local:
                    try:
                        ip = ipaddress.ip_address(parsed.hostname)
                        if ip.is_private or ip.is_loopback:
                            raise SecurityError(f"Local/private IP not allowed in URL: {parsed.hostname}")
                    except ValueError:
                        # Not an IP, that's fine
                        pass
            
            return url
            
        except Exception as e:
            raise ValidationError(f"Invalid URL: {str(e)}")
    
    def detect_suspicious_content(self, text: str) -> List[str]:
        """Detect suspicious content in text."""
        suspicious_findings = []
        text_lower = text.lower()
        
        # Check for suspicious keywords
        for keyword in self.SUSPICIOUS_KEYWORDS:
            if keyword in text_lower:
                suspicious_findings.append(f"Suspicious keyword: {keyword}")
        
        # Check for patterns that might indicate attempts to bypass validation
        bypass_patterns = [
            r'<!--.*?-->',  # HTML comments
            r'/\*.*?\*/',   # CSS/JS comments
            r'\\u[0-9a-fA-F]{4}',  # Unicode escapes
            r'\\x[0-9a-fA-F]{2}',  # Hex escapes
            r'%[0-9a-fA-F]{2}',    # URL encoding
        ]
        
        for pattern in bypass_patterns:
            if re.search(pattern, text, re.DOTALL):
                suspicious_findings.append(f"Potential bypass attempt: {pattern}")
        
        return suspicious_findings
    
    def calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content for integrity checking."""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()
    
    def validate_content_length(self, content: str, max_length: Optional[int] = None) -> None:
        """Validate content length with configurable limits."""
        max_len = max_length or self.config.max_string_length
        if len(content) > max_len:
            raise ValidationError(
                f"Content too large: {len(content)} > {max_len}",
                {"actual_length": len(content), "max_length": max_len}
            )
    
    def validate_string_advanced(
        self, 
        value: str, 
        field_name: str, 
        min_length: int = 0, 
        max_length: Optional[int] = None,
        allow_empty: bool = False,
        check_suspicious: bool = True
    ) -> str:
        """Enhanced string validation with security features."""
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", {
                "field": field_name,
                "type": type(value).__name__
            })
        
        if not allow_empty and not value.strip():
            raise ValidationError(f"{field_name} cannot be empty", {
                "field": field_name
            })
        
        # Use configurable max length
        max_len = max_length or self.config.max_string_length
        
        if len(value) < min_length:
            raise ValidationError(f"{field_name} must be at least {min_length} characters", {
                "field": field_name,
                "length": len(value),
                "min_length": min_length
            })
        
        if len(value) > max_len:
            raise ValidationError(f"{field_name} must be at most {max_len} characters", {
                "field": field_name,
                "length": len(value),
                "max_length": max_len
            })
        
        # Check for dangerous patterns with caching
        for i, pattern in enumerate(self.DANGEROUS_PATTERNS):
            pattern_key = f"{field_name}:{i}:{hashlib.md5(value.encode()).hexdigest()[:8]}"
            
            # Use cache to avoid recomputing expensive regex matches
            if pattern_key not in self._suspicious_patterns_cache:
                self._suspicious_patterns_cache[pattern_key] = bool(re.search(pattern, value, re.IGNORECASE))
                
                # Limit cache size to prevent memory issues
                if len(self._suspicious_patterns_cache) > 1000:
                    keys_to_remove = list(self._suspicious_patterns_cache.keys())[:200]
                    for key in keys_to_remove:
                        del self._suspicious_patterns_cache[key]
            
            if self._suspicious_patterns_cache[pattern_key]:
                security_logger.log_security_violation(
                    f"Dangerous pattern detected in {field_name}",
                    {
                        "field": field_name,
                        "pattern_index": i,
                        "content_hash": self.calculate_content_hash(value)
                    }
                )
                raise SecurityError(f"Potentially dangerous content detected in {field_name}", {
                    "field": field_name,
                    "pattern_index": i,
                    "content_hash": self.calculate_content_hash(value)
                })
        
        # Check for suspicious content if enabled
        if check_suspicious:
            suspicious_findings = self.detect_suspicious_content(value)
            if suspicious_findings:
                logger.warning(
                    f"Suspicious content detected in {field_name}",
                    extra={
                        "field": field_name,
                        "findings": suspicious_findings,
                        "content_hash": self.calculate_content_hash(value)
                    }
                )
                security_logger.log_suspicious_activity(
                    f"Suspicious content in {field_name}",
                    {
                        "field": field_name,
                        "findings": suspicious_findings
                    }
                )
        
        return value.strip()
    
    @classmethod
    def validate_string(
        cls, 
        value: str, 
        field_name: str, 
        min_length: int = 0, 
        max_length: int = 10000,
        allow_empty: bool = False
    ) -> str:
        """Validate and sanitize string input."""
        if not isinstance(value, str):
            raise ValidationError(f"{field_name} must be a string", {
                "field": field_name,
                "type": type(value).__name__
            })
        
        if not allow_empty and not value.strip():
            raise ValidationError(f"{field_name} cannot be empty", {
                "field": field_name
            })
        
        if len(value) < min_length:
            raise ValidationError(f"{field_name} must be at least {min_length} characters", {
                "field": field_name,
                "length": len(value),
                "min_length": min_length
            })
        
        if len(value) > max_length:
            raise ValidationError(f"{field_name} must be at most {max_length} characters", {
                "field": field_name,
                "length": len(value),
                "max_length": max_length
            })
        
        # Check for dangerous patterns
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, value, re.IGNORECASE):
                raise SecurityError(f"Potentially dangerous content detected in {field_name}", {
                    "field": field_name,
                    "pattern": pattern
                })
        
        return value.strip()
    
    @classmethod
    def validate_model_name(cls, name: str) -> str:
        """Validate model name."""
        name = cls.validate_string(name, "model_name", min_length=1, max_length=100)
        
        # Allow alphanumeric, hyphens, underscores, and dots
        if not re.match(r'^[a-zA-Z0-9._-]+$', name):
            raise ValidationError("Model name can only contain letters, numbers, dots, hyphens, and underscores", {
                "field": "model_name",
                "value": name
            })
        
        return name
    
    @classmethod
    def validate_provider(cls, provider: str) -> str:
        """Validate model provider."""
        provider = cls.validate_string(provider, "provider", min_length=1, max_length=50)
        
        allowed_providers = ["openai", "anthropic", "local", "huggingface", "google"]
        if provider.lower() not in allowed_providers:
            raise ValidationError(f"Provider must be one of: {', '.join(allowed_providers)}", {
                "field": "provider",
                "value": provider,
                "allowed": allowed_providers
            })
        
        return provider.lower()
    
    @classmethod
    def validate_api_key(cls, api_key: Optional[str], provider: str) -> Optional[str]:
        """Validate API key format."""
        if not api_key:
            return None
        
        api_key = cls.validate_string(api_key, "api_key", min_length=10, max_length=200)
        
        # Basic format validation for known providers
        if provider == "openai" and not api_key.startswith("sk-"):
            raise ValidationError("OpenAI API key must start with 'sk-'", {
                "field": "api_key",
                "provider": provider
            })
        
        if provider == "anthropic" and not api_key.startswith("sk-ant-"):
            raise ValidationError("Anthropic API key must start with 'sk-ant-'", {
                "field": "api_key", 
                "provider": provider
            })
        
        return api_key
    
    @classmethod
    def validate_temperature(cls, temperature: float) -> float:
        """Validate temperature parameter."""
        if not isinstance(temperature, (int, float)):
            raise ValidationError("Temperature must be a number", {
                "field": "temperature",
                "type": type(temperature).__name__
            })
        
        if not 0.0 <= temperature <= 2.0:
            raise ValidationError("Temperature must be between 0.0 and 2.0", {
                "field": "temperature",
                "value": temperature
            })
        
        return float(temperature)
    
    @classmethod
    def validate_max_tokens(cls, max_tokens: int) -> int:
        """Validate max tokens parameter."""
        if not isinstance(max_tokens, int):
            raise ValidationError("Max tokens must be an integer", {
                "field": "max_tokens",
                "type": type(max_tokens).__name__
            })
        
        if not 1 <= max_tokens <= 8192:
            raise ValidationError("Max tokens must be between 1 and 8192", {
                "field": "max_tokens",
                "value": max_tokens
            })
        
        return max_tokens
    
    @classmethod
    def validate_benchmarks(cls, benchmarks: Union[str, List[str]]) -> List[str]:
        """Validate benchmark names."""
        if isinstance(benchmarks, str):
            if benchmarks.lower() == "all":
                return ["all"]
            benchmarks = [benchmarks]
        
        if not isinstance(benchmarks, list):
            raise ValidationError("Benchmarks must be a string or list of strings", {
                "field": "benchmarks",
                "type": type(benchmarks).__name__
            })
        
        if not benchmarks:
            raise ValidationError("At least one benchmark must be specified", {
                "field": "benchmarks"
            })
        
        validated_benchmarks = []
        allowed_benchmarks = ["all", "truthfulqa", "mmlu", "humaneval"]
        
        for benchmark in benchmarks:
            if not isinstance(benchmark, str):
                raise ValidationError("Each benchmark must be a string", {
                    "field": "benchmarks",
                    "type": type(benchmark).__name__
                })
            
            benchmark = benchmark.lower().strip()
            if benchmark not in allowed_benchmarks:
                raise ValidationError(f"Unknown benchmark: {benchmark}", {
                    "field": "benchmarks",
                    "value": benchmark,
                    "allowed": allowed_benchmarks
                })
            
            validated_benchmarks.append(benchmark)
        
        return validated_benchmarks
    
    @classmethod
    def validate_file_path(cls, path: str, must_exist: bool = False) -> Path:
        """Validate file path."""
        path_str = cls.validate_string(path, "file_path", min_length=1, max_length=500)
        
        # Prevent path traversal attacks
        if ".." in path_str or path_str.startswith("/"):
            raise SecurityError("Path traversal detected", {
                "field": "file_path",
                "value": path_str
            })
        
        path_obj = Path(path_str)
        
        if must_exist and not path_obj.exists():
            raise ValidationError(f"File does not exist: {path_str}", {
                "field": "file_path",
                "path": path_str
            })
        
        return path_obj
    
    @classmethod
    def validate_json_data(cls, data: str, field_name: str) -> Dict[str, Any]:
        """Validate and parse JSON data."""
        data = cls.validate_string(data, field_name, max_length=100000)
        
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError as e:
            raise ValidationError(f"Invalid JSON in {field_name}: {str(e)}", {
                "field": field_name,
                "error": str(e)
            })
        
        if not isinstance(parsed, dict):
            raise ValidationError(f"{field_name} must be a JSON object", {
                "field": field_name,
                "type": type(parsed).__name__
            })
        
        return parsed
    
    @classmethod
    def sanitize_user_input(cls, text: str) -> str:
        """Sanitize user input by removing potentially dangerous content."""
        if not isinstance(text, str):
            return str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove script content
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
        
        # Remove JavaScript protocols
        text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
        text = re.sub(r'vbscript:', '', text, flags=re.IGNORECASE)
        
        # Remove event handlers
        text = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', text, flags=re.IGNORECASE)
        
        # Limit length to prevent DoS
        if len(text) > 10000:
            text = text[:10000] + "..."
        
        return text.strip()


class ResourceValidator:
    """Validate resource constraints."""
    
    @classmethod
    def validate_memory_usage(cls, current_mb: int, limit_mb: int = 1024) -> None:
        """Validate memory usage doesn't exceed limits."""
        if current_mb > limit_mb:
            raise ResourceError(f"Memory usage ({current_mb}MB) exceeds limit ({limit_mb}MB)", {
                "current": current_mb,
                "limit": limit_mb
            })
    
    @classmethod
    def validate_disk_space(cls, required_mb: int, available_mb: int) -> None:
        """Validate sufficient disk space is available."""
        if required_mb > available_mb:
            raise ResourceError(f"Insufficient disk space. Required: {required_mb}MB, Available: {available_mb}MB", {
                "required": required_mb,
                "available": available_mb
            })
    
    @classmethod
    def validate_concurrent_jobs(cls, current_jobs: int, max_jobs: int = 10) -> None:
        """Validate number of concurrent jobs doesn't exceed limits."""
        if current_jobs >= max_jobs:
            raise ResourceError(f"Maximum concurrent jobs ({max_jobs}) reached. Current: {current_jobs}", {
                "current": current_jobs,
                "max": max_jobs
            })