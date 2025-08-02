# Build & Deployment Guide

## Overview

This guide covers building, containerizing, and deploying the AGI Evaluation Sandbox across different environments.

## Quick Start

```bash
# Local development
make setup
make dev

# Docker development
make docker-run

# Production build
make build
make docker-build
```

## Build System

### Architecture

The build system uses a multi-stage approach:
1. **Development**: Fast iteration with hot reloading
2. **Testing**: Isolated test environments with all dependencies
3. **Production**: Optimized, secure, minimal containers

### Build Tools

- **Python**: Poetry/pip for dependency management, setuptools for packaging
- **JavaScript/TypeScript**: npm/yarn for dependencies, Vite/webpack for bundling
- **Docker**: Multi-stage builds for optimization
- **Make**: Unified command interface across platforms

## Local Development

### Prerequisites

- Python 3.11+
- Node.js 18+
- Docker & Docker Compose
- Make (optional but recommended)

### Environment Setup

```bash
# Clone repository
git clone https://github.com/danieleschmidt/agi-eval-sandbox.git
cd agi-eval-sandbox

# Setup development environment
make setup

# Start development servers
make dev
```

### Manual Setup (without Make)

```bash
# Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Node.js dependencies
npm install

# Pre-commit hooks
pre-commit install

# Environment variables
cp .env.example .env
# Edit .env with your configuration
```

## Build Processes

### Python API Build

```bash
# Development
cd api
pip install -e ".[dev]"
uvicorn main:app --reload

# Production build
python -m build
pip install dist/*.whl
```

### Dashboard Build

```bash
# Development
cd dashboard
npm install
npm run dev

# Production build
npm run build
# Output in dashboard/dist/
```

### Unified Build

```bash
# Build all components
make build

# Build specific components
make build-api
make build-dashboard
```

## Docker Containerization

### Multi-Stage Dockerfile

```dockerfile
# Build stage for Python
FROM python:3.11-slim as python-builder
# ... build Python app

# Build stage for Node.js
FROM node:18-alpine as node-builder
# ... build dashboard

# Production stage
FROM python:3.11-slim as production
# ... combine artifacts
```

### Docker Commands

```bash
# Build image
docker build -t agi-eval-sandbox:latest .

# Build with specific target
docker build --target python-builder -t agi-eval-sandbox:api .

# Run container
docker run -p 8000:8000 -p 8080:8080 agi-eval-sandbox:latest

# Multi-platform build
docker buildx build --platform linux/amd64,linux/arm64 -t agi-eval-sandbox:latest .
```

### Docker Compose

```bash
# Development environment
docker-compose -f docker-compose.dev.yml up

# Production environment
docker-compose up --build

# Testing environment
docker-compose -f docker-compose.test.yml up --abort-on-container-exit
```

## Testing Builds

### Automated Testing

```bash
# Run all tests
make test

# Test in Docker
make docker-test

# Specific test types
make test-api
make test-dashboard
make test-e2e
```

### Test Environment

The testing environment includes:
- Isolated test databases
- Mock external services
- Performance testing tools
- Security scanning
- Code coverage reporting

```yaml
# docker-compose.test.yml structure
services:
  postgres-test:     # Test database
  redis-test:        # Test cache
  api-test:          # API tests with coverage
  dashboard-test:    # Frontend tests
  e2e-test:          # End-to-end tests
  security-test:     # Security scans
  performance-test:  # Load testing
```

## Production Deployment

### Container Registry

```bash
# Tag for registry
docker tag agi-eval-sandbox:latest your-registry.com/agi-eval-sandbox:v1.0.0

# Push to registry
docker push your-registry.com/agi-eval-sandbox:v1.0.0

# Multi-architecture push
docker buildx build --platform linux/amd64,linux/arm64 \
  --push -t your-registry.com/agi-eval-sandbox:v1.0.0 .
```

### Environment Configuration

#### Development
```bash
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
DATABASE_URL=postgresql://user:pass@localhost:5432/agi_eval_dev
```

#### Staging
```bash
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO
DATABASE_URL=postgresql://user:pass@staging-db:5432/agi_eval_staging
```

#### Production
```bash
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING
DATABASE_URL=postgresql://user:pass@prod-db:5432/agi_eval_prod
SENTRY_DSN=https://...
```

### Kubernetes Deployment

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: agi-eval-sandbox
spec:
  replicas: 3
  selector:
    matchLabels:
      app: agi-eval-sandbox
  template:
    metadata:
      labels:
        app: agi-eval-sandbox
    spec:
      containers:
      - name: app
        image: your-registry.com/agi-eval-sandbox:v1.0.0
        ports:
        - containerPort: 8000
        - containerPort: 8080
        env:
        - name: ENVIRONMENT
          value: "production"
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: database-secret
              key: url
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/build.yml
name: Build and Test
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: make docker-test
      
  build:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v3
      - name: Build Docker image
        run: make docker-build
      
  deploy:
    runs-on: ubuntu-latest
    needs: build
    if: github.ref == 'refs/heads/main'
    steps:
      - name: Deploy to staging
        run: make deploy-staging
```

### GitLab CI

```yaml
# .gitlab-ci.yml
stages:
  - test
  - build
  - deploy

test:
  stage: test
  script:
    - make docker-test
  artifacts:
    reports:
      junit: test-results/*.xml
      coverage: coverage.xml

build:
  stage: build
  script:
    - make docker-build
    - docker tag agi-eval-sandbox:latest $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA
    - docker push $CI_REGISTRY_IMAGE:$CI_COMMIT_SHA

deploy_staging:
  stage: deploy
  script:
    - make deploy-staging
  only:
    - develop

deploy_production:
  stage: deploy
  script:
    - make deploy-prod
  only:
    - main
  when: manual
```

## Performance Optimization

### Build Optimization

```dockerfile
# Use build cache
FROM python:3.11-slim as base
COPY requirements.txt .
RUN pip install -r requirements.txt

# Multi-stage builds
FROM base as development
COPY requirements-dev.txt .
RUN pip install -r requirements-dev.txt

FROM base as production
# Minimal production dependencies only
```

### Image Optimization

```bash
# Use .dockerignore to reduce context
echo "tests/" >> .dockerignore
echo "docs/" >> .dockerignore

# Squash layers
docker build --squash -t agi-eval-sandbox:latest .

# Analyze image size
docker images agi-eval-sandbox:latest
dive agi-eval-sandbox:latest
```

### Runtime Optimization

```yaml
# docker-compose.yml optimizations
services:
  app:
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
    healthcheck:
      test: ["CMD", "python", "/healthcheck.py"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

## Security Considerations

### Container Security

```dockerfile
# Use non-root user
RUN groupadd -r agi_eval && useradd -r -g agi_eval agi_eval
USER agi_eval

# Minimal base image
FROM python:3.11-slim
# or distroless
FROM gcr.io/distroless/python3

# Security scanning
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*
```

### Secrets Management

```bash
# Use Docker secrets
echo "database_password" | docker secret create db_pass -

# Environment-specific secrets
kubectl create secret generic app-secrets \
  --from-literal=database-url="postgresql://..." \
  --from-literal=api-key="..."
```

### Network Security

```yaml
# docker-compose.yml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true  # No external access

services:
  app:
    networks:
      - frontend
      - backend
  database:
    networks:
      - backend  # Only backend access
```

## Monitoring & Observability

### Build Metrics

```bash
# Build time tracking
time make build

# Image size analysis
docker images --format "table {{.Repository}}\\t{{.Tag}}\\t{{.Size}}"

# Build cache analysis
docker builder prune --filter until=24h
```

### Runtime Metrics

```yaml
# Prometheus metrics in docker-compose.yml
services:
  app:
    labels:
      - "prometheus.io/scrape=true"
      - "prometheus.io/port=8000"
      - "prometheus.io/path=/metrics"
```

## Troubleshooting

### Common Build Issues

#### Python Dependencies
```bash
# Clear pip cache
pip cache purge

# Reinstall from scratch
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Node.js Dependencies
```bash
# Clear npm cache
npm cache clean --force

# Delete node_modules
rm -rf node_modules package-lock.json
npm install
```

#### Docker Issues
```bash
# Clear Docker cache
docker system prune -a

# Rebuild without cache
docker build --no-cache -t agi-eval-sandbox:latest .

# Check disk space
docker system df
```

### Build Debugging

```bash
# Verbose build output
make build V=1

# Docker build with progress
docker build --progress=plain -t agi-eval-sandbox:latest .

# Debug failed container
docker run -it --entrypoint /bin/bash agi-eval-sandbox:latest
```

### Performance Issues

```bash
# Profile build time
time make build

# Analyze layer sizes
dive agi-eval-sandbox:latest

# Check resource usage
docker stats
```

## Advanced Topics

### Multi-Architecture Builds

```bash
# Setup buildx
docker buildx create --name multiarch --use
docker buildx inspect --bootstrap

# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64,linux/arm/v7 \
  --push \
  -t your-registry.com/agi-eval-sandbox:v1.0.0 .
```

### Automated Security Scanning

```yaml
# .github/workflows/security.yml
name: Security Scan
on: [push]

jobs:
  security:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run Trivy vulnerability scanner
        uses: aquasecurity/trivy-action@master
        with:
          image-ref: 'agi-eval-sandbox:latest'
          format: 'sarif'
          output: 'trivy-results.sarif'
      - name: Upload Trivy scan results
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: 'trivy-results.sarif'
```

### Custom Build Environments

```dockerfile
# Dockerfile.dev - Development with debugging tools
FROM agi-eval-sandbox:latest as dev
RUN pip install debugpy ipdb
EXPOSE 5678
CMD ["python", "-m", "debugpy", "--listen", "0.0.0.0:5678", "--wait-for-client", "-m", "uvicorn", "main:app"]
```

This comprehensive build guide ensures consistent, secure, and optimized builds across all environments.