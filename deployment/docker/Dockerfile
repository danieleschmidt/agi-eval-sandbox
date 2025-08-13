# Multi-stage production Dockerfile for AGI Evaluation Sandbox
FROM python:3.12-slim as builder

# Build arguments
ARG ENVIRONMENT=production
ARG BUILD_DATE
ARG VERSION=1.0.0
ARG VCS_REF

# Labels for container metadata
LABEL maintainer="Terragon Labs <contact@terragonlabs.com>" \
      version="${VERSION}" \
      description="AGI Evaluation Sandbox - Production-ready AI model evaluation platform" \
      build-date="${BUILD_DATE}" \
      vcs-ref="${VCS_REF}" \
      environment="${ENVIRONMENT}"

# Security: Create non-root user
RUN groupadd -r agi_eval && useradd -r -g agi_eval -d /app -s /bin/bash agi_eval

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Production stage
FROM python:3.12-slim as production

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Create non-root user
RUN groupadd -r agi_eval && useradd -r -g agi_eval -d /app -s /bin/bash agi_eval

# Set working directory
WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY --chown=agi_eval:agi_eval src/ ./src/
COPY --chown=agi_eval:agi_eval deployment/config/ ./config/
COPY --chown=agi_eval:agi_eval deployment/scripts/ ./scripts/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/cache && \
    chown -R agi_eval:agi_eval /app

# Environment variables
ENV PYTHONPATH=/app \
    ENVIRONMENT=production \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Security: Switch to non-root user
USER agi_eval

# Expose ports
EXPOSE 8080

# Default command
CMD ["python", "-m", "src.agi_eval_sandbox.api.main"]