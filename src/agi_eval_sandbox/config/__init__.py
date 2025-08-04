"""Configuration module for AGI Evaluation Sandbox."""

try:
    from .settings import settings
except ImportError:
    # Fallback to simple settings without pydantic
    from .simple_settings import settings

__all__ = ["settings"]