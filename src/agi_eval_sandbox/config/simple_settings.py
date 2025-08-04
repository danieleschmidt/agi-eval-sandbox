"""Simple settings without external dependencies."""

import os


class SimpleSettings:
    """Simple settings class without pydantic dependency."""
    
    def __init__(self):
        # Application
        self.app_name = os.getenv("APP_NAME", "AGI Evaluation Sandbox")
        self.debug = os.getenv("DEBUG", "False").lower() == "true"
        self.version = os.getenv("VERSION", "0.1.0")
        
        # API Configuration
        self.api_host = os.getenv("API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("API_PORT", "8080"))
        self.api_workers = int(os.getenv("API_WORKERS", "4"))
        
        # Database
        self.database_url = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/agi_eval")
        self.database_echo = os.getenv("DATABASE_ECHO", "False").lower() == "true"
        
        # Redis
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # Celery
        self.celery_broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
        self.celery_result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
        
        # Model Providers
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        
        # Storage
        self.storage_backend = os.getenv("STORAGE_BACKEND", "local")
        self.storage_path = os.getenv("STORAGE_PATH", "./data")
        self.s3_bucket = os.getenv("S3_BUCKET")
        self.s3_region = os.getenv("S3_REGION")
        
        # Security
        self.secret_key = os.getenv("SECRET_KEY", "dev-secret-key")
        self.access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        
        # Monitoring
        self.prometheus_enabled = os.getenv("PROMETHEUS_ENABLED", "True").lower() == "true"
        self.jaeger_enabled = os.getenv("JAEGER_ENABLED", "False").lower() == "true"
        self.jaeger_agent_host = os.getenv("JAEGER_AGENT_HOST", "localhost")
        self.jaeger_agent_port = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
        
        # Performance
        self.max_concurrent_evaluations = int(os.getenv("MAX_CONCURRENT_EVALUATIONS", "10"))
        self.evaluation_timeout = int(os.getenv("EVALUATION_TIMEOUT", "3600"))


# Global settings instance
settings = SimpleSettings()