"""Testing utilities and helper functions."""

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional
from unittest.mock import patch

def load_fixture(filename: str) -> Any:
    """Load a test fixture file."""
    fixture_path = Path(__file__).parent / "fixtures" / "data" / filename
    
    if not fixture_path.exists():
        raise FileNotFoundError(f"Fixture file not found: {fixture_path}")
    
    with open(fixture_path, 'r') as f:
        if filename.endswith('.json'):
            return json.load(f)
        else:
            return f.read()

def create_temp_config(config: Dict[str, Any]) -> str:
    """Create a temporary configuration file."""
    with tempfile.NamedTemporaryFile(
        mode='w', 
        suffix='.json', 
        delete=False
    ) as f:
        json.dump(config, f, indent=2)
        return f.name

@contextmanager
def temp_env_vars(**env_vars) -> Generator[None, None, None]:
    """Temporarily set environment variables."""
    old_env = {}
    
    # Set new environment variables
    for key, value in env_vars.items():
        old_env[key] = os.environ.get(key)
        os.environ[key] = str(value)
    
    try:
        yield
    finally:
        # Restore original environment
        for key, old_value in old_env.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value

@contextmanager
def mock_api_responses(responses: Dict[str, Any]) -> Generator[None, None, None]:
    """Mock API responses for testing."""
    with patch('requests.get') as mock_get, \
         patch('requests.post') as mock_post:
        
        def get_response(url, **kwargs):
            if url in responses:
                mock_response = type('MockResponse', (), {
                    'json': lambda: responses[url],
                    'status_code': 200,
                    'text': json.dumps(responses[url])
                })()
                return mock_response
            return None
        
        mock_get.side_effect = get_response
        mock_post.side_effect = get_response
        
        yield

def assert_dict_contains(expected: Dict[str, Any], actual: Dict[str, Any]) -> None:
    """Assert that actual dict contains all key-value pairs from expected dict."""
    for key, expected_value in expected.items():
        assert key in actual, f"Key '{key}' not found in actual dict"
        assert actual[key] == expected_value, \
            f"Value mismatch for key '{key}': expected {expected_value}, got {actual[key]}"

def assert_list_contains_items(expected_items: list, actual_list: list) -> None:
    """Assert that actual list contains all items from expected list."""
    for item in expected_items:
        assert item in actual_list, f"Item '{item}' not found in actual list"

class TestDataBuilder:
    """Builder pattern for creating test data."""
    
    def __init__(self):
        self.data = {}
    
    def with_field(self, key: str, value: Any) -> 'TestDataBuilder':
        """Add a field to the test data."""
        self.data[key] = value
        return self
    
    def with_nested_field(self, path: str, value: Any) -> 'TestDataBuilder':
        """Add a nested field using dot notation (e.g., 'config.model.name')."""
        keys = path.split('.')
        current = self.data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
        return self
    
    def build(self) -> Dict[str, Any]:
        """Build and return the test data."""
        return self.data.copy()

def create_benchmark_config(
    name: str = "test-benchmark",
    questions_count: int = 10,
    **kwargs
) -> Dict[str, Any]:
    """Create a test benchmark configuration."""
    config = {
        "name": name,
        "questions_count": questions_count,
        "timeout": 30,
        "max_retries": 3,
        "scoring": "accuracy"
    }
    config.update(kwargs)
    return config

def create_model_config(
    provider: str = "mock",
    name: str = "test-model",
    **kwargs
) -> Dict[str, Any]:
    """Create a test model configuration."""
    config = {
        "provider": provider,
        "name": name,
        "temperature": 0.0,
        "max_tokens": 2048,
        "timeout": 30
    }
    config.update(kwargs)
    return config

def skip_if_no_api_key(api_key_env_var: str):
    """Decorator to skip tests if API key is not available."""
    import pytest
    
    def decorator(test_func):
        return pytest.mark.skipif(
            not os.environ.get(api_key_env_var),
            reason=f"API key {api_key_env_var} not available"
        )(test_func)
    
    return decorator

def parametrize_models(*model_configs):
    """Decorator to parametrize tests with different model configurations."""
    import pytest
    
    return pytest.mark.parametrize(
        "model_config", 
        model_configs,
        ids=[f"{config['provider']}-{config['name']}" for config in model_configs]
    )