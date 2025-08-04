"""Input validation and sanitization utilities."""

import re
import json
from typing import Any, Dict, List, Optional, Union
from pathlib import Path

from .exceptions import ValidationError, SecurityError


class InputValidator:
    """Comprehensive input validation."""
    
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
    ]
    
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