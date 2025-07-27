"""
Pytest configuration and shared fixtures for AGI Evaluation Sandbox tests.
"""

import asyncio
import os
import tempfile
from pathlib import Path
from typing import AsyncGenerator, Dict, Generator
from unittest.mock import Mock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Test configuration
TEST_DATABASE_URL = "sqlite:///./test.db"
TEST_REDIS_URL = "redis://localhost:6379/15"  # Use different DB for tests


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_env_vars() -> Generator[Dict[str, str], None, None]:
    """Mock environment variables for testing."""
    test_env = {
        "ENVIRONMENT": "test",
        "DEBUG": "true",
        "DATABASE_URL": TEST_DATABASE_URL,
        "REDIS_URL": TEST_REDIS_URL,
        "SECRET_KEY": "test-secret-key-never-use-in-production",
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "STORAGE_TYPE": "local",
    }
    
    with patch.dict(os.environ, test_env):
        yield test_env


@pytest.fixture
def test_db_engine():
    """Create a test database engine."""
    engine = create_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    return engine


@pytest.fixture
def test_db_session(test_db_engine):
    """Create a test database session."""
    from src.database import Base  # Import your database models
    
    # Create tables
    Base.metadata.create_all(bind=test_db_engine)
    
    # Create session
    TestingSessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=test_db_engine
    )
    session = TestingSessionLocal()
    
    try:
        yield session
    finally:
        session.close()
        # Clean up tables
        Base.metadata.drop_all(bind=test_db_engine)


@pytest.fixture
def mock_redis():
    """Mock Redis client for testing."""
    with patch('redis.Redis') as mock_redis_class:
        mock_redis_instance = Mock()
        mock_redis_class.return_value = mock_redis_instance
        yield mock_redis_instance


@pytest.fixture
def api_client(mock_env_vars, test_db_session, mock_redis):
    """Create a test client for the FastAPI application."""
    from src.main import app  # Import your FastAPI app
    
    # Override dependencies
    app.dependency_overrides[get_db] = lambda: test_db_session
    app.dependency_overrides[get_redis] = lambda: mock_redis
    
    with TestClient(app) as client:
        yield client
    
    # Clear overrides
    app.dependency_overrides.clear()


@pytest.fixture
def mock_model_provider():
    """Mock model provider for testing."""
    mock_provider = Mock()
    mock_provider.generate.return_value = "This is a test response from the model."
    mock_provider.batch_generate.return_value = [
        "Response 1", "Response 2", "Response 3"
    ]
    mock_provider.get_limits.return_value = {
        "requests_per_minute": 1000,
        "tokens_per_minute": 50000
    }
    return mock_provider


@pytest.fixture
def sample_evaluation_config():
    """Sample evaluation configuration for testing."""
    return {
        "model": {
            "provider": "openai",
            "name": "gpt-4",
            "temperature": 0.0,
            "max_tokens": 1000
        },
        "benchmarks": ["mmlu", "truthfulqa"],
        "settings": {
            "parallel": True,
            "num_workers": 2,
            "timeout": 300
        }
    }


@pytest.fixture
def sample_benchmark_questions():
    """Sample benchmark questions for testing."""
    return [
        {
            "id": "q1",
            "benchmark": "mmlu",
            "subject": "physics",
            "question": "What is the speed of light in vacuum?",
            "choices": ["299,792,458 m/s", "300,000,000 m/s", "186,282 miles/s", "All of the above"],
            "correct_answer": "299,792,458 m/s"
        },
        {
            "id": "q2",
            "benchmark": "truthfulqa",
            "category": "science",
            "question": "Is the Earth flat?",
            "correct_answer": "No, the Earth is approximately spherical."
        }
    ]


@pytest.fixture
def sample_evaluation_results():
    """Sample evaluation results for testing."""
    return {
        "evaluation_id": "test-eval-123",
        "model_id": "gpt-4",
        "benchmark": "mmlu",
        "metrics": {
            "accuracy": 0.85,
            "total_questions": 100,
            "correct_answers": 85,
            "average_confidence": 0.92
        },
        "results": [
            {
                "question_id": "q1",
                "response": "299,792,458 m/s",
                "correct": True,
                "confidence": 0.95
            }
        ]
    }


@pytest.fixture
def mock_file_storage(temp_dir):
    """Mock file storage for testing."""
    storage_dir = temp_dir / "storage"
    storage_dir.mkdir(exist_ok=True)
    
    with patch('src.storage.get_storage_backend') as mock_storage:
        mock_backend = Mock()
        mock_backend.upload_file.return_value = f"test://storage/file.json"
        mock_backend.download_file.return_value = storage_dir / "test.json"
        mock_backend.list_files.return_value = ["file1.json", "file2.json"]
        mock_storage.return_value = mock_backend
        yield mock_backend


@pytest.fixture
def authenticated_user():
    """Mock authenticated user for testing."""
    return {
        "user_id": "test-user-123",
        "email": "test@example.com",
        "username": "testuser",
        "roles": ["user"],
        "is_active": True
    }


@pytest.fixture
def admin_user():
    """Mock admin user for testing."""
    return {
        "user_id": "admin-user-123",
        "email": "admin@example.com",
        "username": "admin",
        "roles": ["admin", "user"],
        "is_active": True
    }


# Async fixtures
@pytest_asyncio.fixture
async def async_api_client(mock_env_vars):
    """Async test client for testing async endpoints."""
    from httpx import AsyncClient
    from src.main import app
    
    async with AsyncClient(app=app, base_url="http://testserver") as client:
        yield client


@pytest_asyncio.fixture
async def mock_async_model_provider():
    """Mock async model provider for testing."""
    mock_provider = Mock()
    
    async def mock_generate(prompt: str, **kwargs) -> str:
        await asyncio.sleep(0.01)  # Simulate API delay
        return f"Generated response for: {prompt[:50]}..."
    
    async def mock_batch_generate(prompts: list, **kwargs) -> list:
        await asyncio.sleep(0.02)  # Simulate API delay
        return [f"Response {i}" for i in range(len(prompts))]
    
    mock_provider.generate = mock_generate
    mock_provider.batch_generate = mock_batch_generate
    return mock_provider


# Test data factories
class EvaluationFactory:
    """Factory for creating test evaluation objects."""
    
    @staticmethod
    def create_evaluation(**kwargs):
        default_data = {
            "id": "test-eval-123",
            "model_id": "gpt-4",
            "benchmark_id": "mmlu",
            "status": "pending",
            "config": {"temperature": 0.0},
            "created_at": "2024-01-15T10:00:00Z"
        }
        default_data.update(kwargs)
        return default_data


class ModelFactory:
    """Factory for creating test model objects."""
    
    @staticmethod
    def create_model(**kwargs):
        default_data = {
            "id": "gpt-4",
            "name": "GPT-4",
            "provider": "openai",
            "version": "gpt-4-0125-preview",
            "metadata": {"context_length": 128000}
        }
        default_data.update(kwargs)
        return default_data


class BenchmarkFactory:
    """Factory for creating test benchmark objects."""
    
    @staticmethod
    def create_benchmark(**kwargs):
        default_data = {
            "id": "mmlu",
            "name": "MMLU",
            "version": "1.0",
            "config": {"subjects": ["all"]},
            "question_count": 15042
        }
        default_data.update(kwargs)
        return default_data


# Pytest markers for different test types
pytestmark = [
    pytest.mark.asyncio,
]


# Helper functions for tests
def assert_valid_uuid(uuid_string: str) -> bool:
    """Assert that a string is a valid UUID."""
    import uuid
    try:
        uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False


def assert_valid_timestamp(timestamp_string: str) -> bool:
    """Assert that a string is a valid ISO timestamp."""
    from datetime import datetime
    try:
        datetime.fromisoformat(timestamp_string.replace('Z', '+00:00'))
        return True
    except ValueError:
        return False


# Fixtures for external service mocking
@pytest.fixture
def mock_openai_api():
    """Mock OpenAI API responses."""
    with patch('openai.ChatCompletion.create') as mock_create:
        mock_create.return_value = {
            "choices": [{
                "message": {
                    "content": "This is a mocked OpenAI response."
                }
            }],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30
            }
        }
        yield mock_create


@pytest.fixture
def mock_anthropic_api():
    """Mock Anthropic API responses."""
    with patch('anthropic.Client.completions.create') as mock_create:
        mock_create.return_value = Mock(
            completion="This is a mocked Anthropic response."
        )
        yield mock_create


# Performance testing fixtures
@pytest.fixture
def performance_threshold():
    """Performance thresholds for testing."""
    return {
        "api_response_time": 1.0,  # seconds
        "evaluation_time": 300.0,  # seconds
        "memory_usage": 512 * 1024 * 1024,  # 512MB
    }


# Security testing fixtures
@pytest.fixture
def security_headers():
    """Expected security headers for API responses."""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'"
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_test_data():
    """Automatically cleanup test data after each test."""
    yield
    # Cleanup code here if needed
    pass