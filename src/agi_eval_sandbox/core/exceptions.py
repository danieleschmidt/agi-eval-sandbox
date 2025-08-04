"""Custom exceptions for AGI Evaluation Sandbox."""

from typing import Optional, Dict, Any


class AGIEvalError(Exception):
    """Base exception for AGI Evaluation Sandbox."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AGIEvalError):
    """Raised when there's a configuration error."""
    pass


class ModelProviderError(AGIEvalError):
    """Raised when there's an error with model providers."""
    pass


class BenchmarkError(AGIEvalError):
    """Raised when there's an error with benchmarks."""
    pass


class EvaluationError(AGIEvalError):
    """Raised when there's an error during evaluation."""
    pass


class ValidationError(AGIEvalError):
    """Raised when input validation fails."""
    pass


class ResourceError(AGIEvalError):
    """Raised when there's a resource-related error (memory, disk, etc.)."""
    pass


class RateLimitError(AGIEvalError):
    """Raised when rate limits are exceeded."""
    pass


class TimeoutError(AGIEvalError):
    """Raised when operations timeout."""
    pass


class SecurityError(AGIEvalError):
    """Raised when security violations are detected."""
    pass


class DataError(AGIEvalError):
    """Raised when there's an error with data processing."""
    pass