# 🚀 AGI Evaluation Sandbox - Production Deployment Guide

This guide provides comprehensive instructions for deploying the AGI Evaluation Sandbox to production environments using Docker Compose or Kubernetes.

## 📋 Table of Contents

- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Docker Deployment](#docker-deployment)
- [Kubernetes Deployment](#kubernetes-deployment)
- [Configuration](#configuration)
- [Monitoring & Observability](#monitoring--observability)
- [Security](#security)
- [Scaling](#scaling)
- [Troubleshooting](#troubleshooting)

## 🔧 Prerequisites

### System Requirements

- **Operating System**: Linux (Ubuntu 20.04+ recommended), macOS, or Windows with WSL2
- **CPU**: 4+ cores recommended
- **Memory**: 8GB+ RAM (16GB+ for production)
- **Storage**: 50GB+ available disk space
- **Network**: Outbound internet access for model API calls

### Software Dependencies

- **Docker**: 20.10+ with Docker Compose v2
- **Kubernetes**: 1.24+ (for K8s deployment)
- **Git**: For cloning the repository
- **OpenSSL**: For generating secrets

## ⚡ Quick Start

1. **Clone the repository**:
   ```bash
   git clone https://github.com/terragonlabs/agi-evaluation-sandbox.git
   cd agi-evaluation-sandbox
   ```

2. **Configure environment**:
   ```bash
   cp deployment/.env.production deployment/.env.local
   # Edit deployment/.env.local with your configuration
   ```

3. **Deploy with Docker**:
   ```bash
   ./deployment/scripts/deploy.sh --build --health-check --report
   ```

4. **Access the services**:
   - API: http://localhost:8080
   - Grafana: http://localhost:3000
   - Prometheus: http://localhost:9090

## 🐳 Docker Deployment

### Architecture Overview

The Docker deployment includes:

- **API Server**: FastAPI application with auto-scaling workers
- **Database**: PostgreSQL 15 with persistent storage
- **Cache**: Redis for caching and message queuing
- **Worker**: Celery workers for background tasks
- **Proxy**: Nginx reverse proxy with SSL termination
- **Monitoring**: Prometheus + Grafana for metrics
- **Tracing**: Jaeger for distributed tracing

### Step-by-Step Deployment

1. **Prepare the environment**:
   ```bash
   # Generate secure passwords
   export SECRET_KEY=$(openssl rand -hex 32)
   export POSTGRES_PASSWORD=$(openssl rand -base64 32)
   export REDIS_PASSWORD=$(openssl rand -base64 32)
   export GRAFANA_PASSWORD="your-secure-password"
   
   # Set API keys (optional but recommended)
   export OPENAI_API_KEY="your-openai-key"
   export ANTHROPIC_API_KEY="your-anthropic-key"
   ```

2. **Build and deploy**:
   ```bash
   cd deployment
   docker-compose up -d --build
   ```

3. **Verify deployment**:
   ```bash
   docker-compose ps
   curl http://localhost:8080/health
   ```

### Service Configuration

#### API Server
- **Port**: 8080
- **Health Check**: `/health`
- **Metrics**: `/metrics`
- **Documentation**: `/docs`

#### Database
- **Engine**: PostgreSQL 15
- **Port**: 5432 (internal)
- **Backup**: Automated daily backups to `/var/lib/postgresql/backups`

#### Monitoring
- **Grafana**: http://localhost:3000 (`admin` / your-password)
- **Prometheus**: http://localhost:9090
- **Jaeger**: http://localhost:16686

## ☸️ Kubernetes Deployment

### Prerequisites

- Kubernetes cluster (1.24+)
- kubectl configured
- cert-manager (for SSL certificates)
- nginx-ingress-controller

### Deployment Steps

1. **Deploy to Kubernetes**:
   ```bash
   ./deployment/scripts/deploy.sh -t kubernetes -n agi-eval --build
   ```

2. **Configure ingress** (optional):
   ```bash
   # Update the ingress host in deployment/kubernetes/deployment.yaml
   kubectl apply -f deployment/kubernetes/
   ```

3. **Monitor deployment**:
   ```bash
   kubectl get pods -n agi-eval
   kubectl logs -f deployment/agi-eval-api -n agi-eval
   ```

### High Availability Configuration

The Kubernetes deployment includes:

- **Horizontal Pod Autoscaler**: 3-20 replicas based on CPU/memory
- **Pod Disruption Budget**: Ensures minimum availability during updates
- **Rolling Updates**: Zero-downtime deployments
- **Health Checks**: Liveness, readiness, and startup probes

## ⚙️ Configuration

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `ENVIRONMENT` | Deployment environment | `production` | No |
| `SECRET_KEY` | Application secret key | Generated | Yes |
| `DATABASE_URL` | PostgreSQL connection string | See env file | Yes |
| `REDIS_URL` | Redis connection string | See env file | Yes |
| `OPENAI_API_KEY` | OpenAI API key | None | No |
| `ANTHROPIC_API_KEY` | Anthropic API key | None | No |
| `GOOGLE_API_KEY` | Google API key | None | No |

### Security Configuration

```bash
# Generate secure secrets
SECRET_KEY=$(openssl rand -hex 32)
POSTGRES_PASSWORD=$(openssl rand -base64 32)
REDIS_PASSWORD=$(openssl rand -base64 32)

# Configure in .env.production
echo "SECRET_KEY=${SECRET_KEY}" >> deployment/.env.production
echo "POSTGRES_PASSWORD=${POSTGRES_PASSWORD}" >> deployment/.env.production
echo "REDIS_PASSWORD=${REDIS_PASSWORD}" >> deployment/.env.production
```

### Performance Tuning

```yaml
# Docker Compose override
services:
  agi-eval-api:
    deploy:
      replicas: 4
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '0.5'
          memory: 1G
```

## 📊 Monitoring & Observability

### Metrics Collection

The deployment includes comprehensive monitoring:

- **Application Metrics**: Request rates, response times, error rates
- **System Metrics**: CPU, memory, disk, network usage
- **Business Metrics**: Evaluation counts, success rates, model performance

### Grafana Dashboards

Pre-configured dashboards for:

1. **System Overview**: Infrastructure health and performance
2. **Application Performance**: API metrics and evaluation statistics
3. **Model Performance**: Benchmark results and comparison metrics
4. **Security Dashboard**: Security events and compliance metrics

### Alerting

Configure alerts for:

- High error rates (>5%)
- Slow response times (>2s)
- High resource usage (>80%)
- Security violations
- Failed evaluations

### Distributed Tracing

Jaeger provides end-to-end request tracing:

- Request flow visualization
- Performance bottleneck identification
- Error root cause analysis
- Dependency mapping

## 🔒 Security

### Security Features

- **Authentication**: JWT-based API authentication
- **Authorization**: Role-based access control
- **Input Validation**: Comprehensive input sanitization
- **Rate Limiting**: Request rate limiting and throttling
- **Security Headers**: HTTPS enforcement, HSTS, CSP
- **Secrets Management**: Encrypted secrets storage
- **Vulnerability Scanning**: Automated security scanning

### Security Hardening

1. **Network Security**:
   ```bash
   # Configure firewall
   sudo ufw allow 80/tcp
   sudo ufw allow 443/tcp
   sudo ufw enable
   ```

2. **Container Security**:
   ```yaml
   security_opt:
     - no-new-privileges:true
   read_only: true
   user: "1000:1000"
   ```

3. **SSL/TLS Configuration**:
   ```bash
   # Generate SSL certificates
   certbot certonly --webroot -w /var/www/html -d your-domain.com
   ```

### Compliance

The system supports compliance with:

- **GDPR**: Data protection and privacy controls
- **SOC 2**: Security controls and audit logging
- **HIPAA**: Healthcare data protection (with additional configuration)

## 📈 Scaling

### Horizontal Scaling

#### Docker Compose
```bash
docker-compose up -d --scale agi-eval-api=4
```

#### Kubernetes
```bash
kubectl scale deployment agi-eval-api --replicas=10 -n agi-eval
```

### Vertical Scaling

Update resource limits:

```yaml
resources:
  requests:
    memory: "1Gi"
    cpu: "500m"
  limits:
    memory: "4Gi" 
    cpu: "2000m"
```

### Auto-Scaling

Kubernetes HPA configuration:

```yaml
metrics:
- type: Resource
  resource:
    name: cpu
    target:
      type: Utilization
      averageUtilization: 70
- type: Resource
  resource:
    name: memory
    target:
      type: Utilization
      averageUtilization: 80
```

## 🔧 Troubleshooting

### Common Issues

#### Container Won't Start
```bash
# Check logs
docker-compose logs agi-eval-api

# Check resource usage
docker stats

# Restart services
docker-compose restart
```

#### Database Connection Issues
```bash
# Test database connectivity
docker-compose exec agi-eval-api python -c "
from src.agi_eval_sandbox.config.settings import settings
print(f'Database URL: {settings.database_url}')
"

# Check PostgreSQL logs
docker-compose logs postgres
```

#### High Memory Usage
```bash
# Monitor memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Scale down if needed
docker-compose up -d --scale agi-eval-api=2
```

### Performance Issues

#### Slow API Responses
1. Check system resources
2. Review database query performance
3. Verify cache hit rates
4. Scale horizontally

#### Failed Evaluations
1. Check model API connectivity
2. Verify API keys and quotas
3. Review rate limiting settings
4. Check evaluation timeouts

### Monitoring & Debugging

#### Health Checks
```bash
# API health
curl http://localhost:8080/health

# System health
curl http://localhost:8080/system/health

# Metrics
curl http://localhost:8080/metrics
```

#### Log Analysis
```bash
# View application logs
docker-compose logs -f agi-eval-api

# Search for errors
docker-compose logs agi-eval-api | grep ERROR

# Tail logs in real-time
docker-compose logs -f --tail=100
```

## 📞 Support

For deployment support and troubleshooting:

- **Documentation**: https://docs.terragonlabs.com/agi-eval
- **Issues**: https://github.com/terragonlabs/agi-evaluation-sandbox/issues
- **Support**: support@terragonlabs.com

## 📄 License

This deployment guide is part of the AGI Evaluation Sandbox project, licensed under the MIT License.