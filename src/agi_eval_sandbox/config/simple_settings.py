"""Simple settings without external dependencies."""

import os


class SimpleSettings:
    """Simple settings class without pydantic dependency."""
    
    def __init__(self):
        # Application
        self.APP_NAME = os.getenv("APP_NAME", "AGI Evaluation Sandbox")
        self.DEBUG = os.getenv("DEBUG", "False").lower() == "true"
        self.VERSION = os.getenv("VERSION", "0.1.0")
        
        # API Configuration
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8080"))
        self.API_WORKERS = int(os.getenv("API_WORKERS", "4"))
        
        # Database
        self.DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/agi_eval")
        self.DATABASE_ECHO = os.getenv("DATABASE_ECHO", "False").lower() == "true"
        
        # Redis
        self.REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        
        # Celery
        self.CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
        self.CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")
        
        # Model Providers
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        self.ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
        self.GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        # Storage
        self.STORAGE_BACKEND = os.getenv("STORAGE_BACKEND", "local")
        self.STORAGE_PATH = os.getenv("STORAGE_PATH", "./data")
        self.S3_BUCKET = os.getenv("S3_BUCKET")
        self.S3_REGION = os.getenv("S3_REGION")
        
        # Security
        self.SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key")
        self.ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
        
        # Monitoring
        self.PROMETHEUS_ENABLED = os.getenv("PROMETHEUS_ENABLED", "True").lower() == "true"
        self.JAEGER_ENABLED = os.getenv("JAEGER_ENABLED", "False").lower() == "true"
        self.JAEGER_AGENT_HOST = os.getenv("JAEGER_AGENT_HOST", "localhost")
        self.JAEGER_AGENT_PORT = int(os.getenv("JAEGER_AGENT_PORT", "6831"))
        
        # Performance
        self.MAX_CONCURRENT_EVALUATIONS = int(os.getenv("MAX_CONCURRENT_EVALUATIONS", "10"))
        self.EVALUATION_TIMEOUT = int(os.getenv("EVALUATION_TIMEOUT", "3600"))
        
        # Logging
        self.LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
        
        # Model Configuration
        self.DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))
        self.DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "1000"))


# Global settings instance
settings = SimpleSettings()