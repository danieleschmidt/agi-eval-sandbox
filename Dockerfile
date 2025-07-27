# AGI Evaluation Sandbox - Multi-stage Docker Build
# =================================================

# Build stage for Python API
FROM python:3.11-slim as python-builder

# Set build arguments
ARG BUILDPLATFORM
ARG TARGETPLATFORM
ARG BUILDARCH
ARG TARGETARCH

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python requirements
COPY api/requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Copy Python source code
COPY api/ ./

# Build Python package
RUN python -m build

# Build stage for Node.js Dashboard
FROM node:18-alpine as node-builder

# Set working directory
WORKDIR /app

# Copy package files
COPY dashboard/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy dashboard source
COPY dashboard/ ./

# Build dashboard
RUN npm run build

# Production stage
FROM python:3.11-slim as production

# Set build metadata
LABEL org.opencontainers.image.title="AGI Evaluation Sandbox"
LABEL org.opencontainers.image.description="One-click evaluation environment for large language models"
LABEL org.opencontainers.image.vendor="Terragon Labs"
LABEL org.opencontainers.image.source="https://github.com/your-org/agi-eval-sandbox"
LABEL org.opencontainers.image.documentation="https://docs.your-org.com/agi-eval"

# Create non-root user
RUN groupadd -r agi_eval && useradd -r -g agi_eval agi_eval

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy Python application from builder
COPY --from=python-builder /app/dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm /tmp/*.whl

# Copy dashboard build from builder
COPY --from=node-builder /app/dist ./static/

# Copy configuration files
COPY docker/entrypoint.sh /entrypoint.sh
COPY docker/healthcheck.py /healthcheck.py

# Make scripts executable
RUN chmod +x /entrypoint.sh /healthcheck.py

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/results && \
    chown -R agi_eval:agi_eval /app

# Switch to non-root user
USER agi_eval

# Expose ports
EXPOSE 8000 8080

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV PORT=8000
ENV DASHBOARD_PORT=8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /healthcheck.py

# Set entrypoint
ENTRYPOINT ["/entrypoint.sh"]

# Default command
CMD ["serve"]