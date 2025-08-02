# Observability Guide

## Overview

The AGI Evaluation Sandbox implements comprehensive observability using the three pillars: metrics, logs, and traces. This guide covers monitoring setup, alerting, and troubleshooting.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │────│ OpenTelemetry   │────│   Jaeger        │
│                 │    │   Collector     │    │   (Traces)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       
         │                       │              ┌─────────────────┐
         │                       └──────────────│   Prometheus    │
         │                                      │   (Metrics)     │
         │                                      └─────────────────┘
         │                                               │
         │              ┌─────────────────┐             │
         └──────────────│  Elasticsearch  │             │
                        │   (Logs)        │             │
                        └─────────────────┘             │
                                 │                      │
                        ┌─────────────────┐    ┌─────────────────┐
                        │     Kibana      │    │    Grafana      │
                        │  (Log Analysis) │    │ (Visualization) │
                        └─────────────────┘    └─────────────────┘
```

## Metrics Collection

### Application Metrics

The application exposes metrics at `/metrics` endpoint using Prometheus format:

#### Business Metrics
- `evaluation_jobs_total` - Total number of evaluation jobs
- `evaluation_jobs_running` - Currently running evaluations
- `evaluation_jobs_completed` - Successfully completed evaluations
- `evaluation_jobs_failed` - Failed evaluations
- `evaluation_duration_seconds` - Time taken for evaluations
- `model_api_requests_total` - API calls to model providers
- `model_api_errors_total` - Failed API calls to model providers

#### HTTP Metrics
- `http_requests_total` - Total HTTP requests
- `http_request_duration_seconds` - Request duration histogram
- `http_requests_in_flight` - Current number of requests being processed

#### System Metrics
- `process_cpu_seconds_total` - CPU time used by the process
- `process_resident_memory_bytes` - Memory usage
- `python_gc_duration_sum` - Garbage collection time

### Infrastructure Metrics

Collected via Prometheus exporters:

#### Node Exporter
- CPU usage, memory, disk, network
- System load and uptime
- File system metrics

#### PostgreSQL Exporter
- Database connections
- Query performance
- Table and index sizes
- Replication lag

#### Redis Exporter
- Memory usage
- Connection count
- Command statistics
- Keyspace information

### Custom Metrics Example

```python
from prometheus_client import Counter, Histogram, Gauge

# Business metrics
EVALUATION_COUNTER = Counter(
    'evaluation_jobs_total',
    'Total number of evaluation jobs',
    ['status', 'model_provider', 'benchmark']
)

EVALUATION_DURATION = Histogram(
    'evaluation_duration_seconds',
    'Time spent on evaluations',
    ['model_provider', 'benchmark'],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1200, 3600)
)

ACTIVE_EVALUATIONS = Gauge(
    'evaluation_jobs_running',
    'Currently running evaluations',
    ['model_provider']
)

# Usage
@app.post("/evaluations")
async def create_evaluation(request: EvaluationRequest):
    start_time = time.time()
    ACTIVE_EVALUATIONS.labels(model_provider=request.model.provider).inc()
    
    try:
        result = await run_evaluation(request)
        EVALUATION_COUNTER.labels(
            status='success',
            model_provider=request.model.provider,
            benchmark=request.benchmark.name
        ).inc()
        return result
    except Exception as e:
        EVALUATION_COUNTER.labels(
            status='failed',
            model_provider=request.model.provider,
            benchmark=request.benchmark.name
        ).inc()
        raise
    finally:
        duration = time.time() - start_time
        EVALUATION_DURATION.labels(
            model_provider=request.model.provider,
            benchmark=request.benchmark.name
        ).observe(duration)
        ACTIVE_EVALUATIONS.labels(model_provider=request.model.provider).dec()
```

## Logging

### Structured Logging

All logs use structured JSON format with consistent fields:

```json
{
  "timestamp": "2024-01-15T10:30:00.123Z",
  "level": "INFO",
  "logger": "agi_eval.evaluation",
  "message": "Evaluation started",
  "correlation_id": "eval-123e4567-e89b-12d3-a456-426614174000",
  "user_id": "user-456",
  "evaluation_id": "eval-789",
  "model_provider": "openai",
  "benchmark": "mmlu",
  "duration_ms": null,
  "status": "started"
}
```

### Log Levels

- **DEBUG**: Detailed information for diagnosing problems
- **INFO**: General information about system operation
- **WARNING**: Something unexpected happened but system continues
- **ERROR**: Serious problem that prevented a function from operating
- **CRITICAL**: Very serious error that may abort the program

### Correlation IDs

Each request gets a unique correlation ID that flows through all logs, enabling request tracing:

```python
import uuid
from contextvars import ContextVar

correlation_id: ContextVar[str] = ContextVar('correlation_id')

@app.middleware("http")
async def add_correlation_id(request: Request, call_next):
    corr_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))
    correlation_id.set(corr_id)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = corr_id
    return response

# Usage in logging
logger.info(
    "Processing evaluation",
    extra={
        "correlation_id": correlation_id.get(),
        "evaluation_id": evaluation.id
    }
)
```

### Log Aggregation

Logs are collected using the ELK stack:

1. **Filebeat** collects logs from containers
2. **Logstash** processes and enriches logs
3. **Elasticsearch** stores and indexes logs
4. **Kibana** provides search and visualization

## Distributed Tracing

### OpenTelemetry Integration

Automatic instrumentation for:
- HTTP requests (FastAPI, requests)
- Database queries (SQLAlchemy, asyncpg)
- Redis operations
- External API calls

```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup tracing
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Manual instrumentation
@tracer.start_as_current_span("evaluate_model")
async def evaluate_model(model: Model, benchmark: Benchmark):
    span = trace.get_current_span()
    span.set_attributes({
        "model.provider": model.provider,
        "model.name": model.name,
        "benchmark.name": benchmark.name,
        "benchmark.type": benchmark.type
    })
    
    try:
        result = await run_evaluation(model, benchmark)
        span.set_attribute("evaluation.score", result.score)
        span.set_status(trace.Status(trace.StatusCode.OK))
        return result
    except Exception as e:
        span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
        span.record_exception(e)
        raise
```

### Trace Analysis

Use Jaeger UI to:
- Track request flows across services
- Identify performance bottlenecks
- Debug failures in distributed operations
- Analyze service dependencies

## Alerting

### Alert Rules

Prometheus alerting rules are defined in `monitoring/alert_rules.yml`:

```yaml
groups:
- name: agi_eval_alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: warning
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} errors per second"

  - alert: EvaluationJobStuck
    expr: evaluation_jobs_running > 0 and increase(evaluation_jobs_completed[30m]) == 0
    for: 30m
    labels:
      severity: critical
    annotations:
      summary: "Evaluation jobs appear to be stuck"
      description: "{{ $value }} jobs running but no completions in 30 minutes"
```

### Notification Channels

#### Slack Integration
```yaml
# alertmanager.yml
route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#agi-eval-alerts'
    title: 'AGI Eval Alert'
    text: '{{ range .Alerts }}{{ .Annotations.description }}{{ end }}'
```

#### Email Notifications
```yaml
receivers:
- name: 'email'
  email_configs:
  - to: 'admin@agi-eval.com'
    from: 'alerts@agi-eval.com'
    subject: 'AGI Eval Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}
```

## Health Checks

### Application Health Endpoints

```python
@app.get("/health")
async def health_check():
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "external_apis": await check_external_apis()
    }
    
    status = "healthy" if all(checks.values()) else "unhealthy"
    
    return {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "version": app.version,
        "checks": checks
    }

@app.get("/ready")
async def readiness_check():
    # Check if app is ready to serve traffic
    return {"status": "ready"}

@app.get("/live")
async def liveness_check():
    # Simple check that app is alive
    return {"status": "alive"}
```

### Container Health Checks

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python /healthcheck.py
```

```python
#!/usr/bin/env python3
# healthcheck.py
import sys
import requests

try:
    response = requests.get("http://localhost:8000/health", timeout=5)
    if response.status_code == 200:
        data = response.json()
        if data.get("status") == "healthy":
            sys.exit(0)
    sys.exit(1)
except Exception:
    sys.exit(1)
```

## Dashboards

### Grafana Dashboards

#### Application Dashboard
- Request rate and response times
- Error rates by endpoint
- Evaluation job status
- Model API usage

#### Infrastructure Dashboard  
- CPU, memory, disk usage
- Network I/O
- Database connections
- Cache hit rates

#### Business Dashboard
- Evaluation success rates
- Model performance trends
- User activity metrics
- Cost tracking

### Dashboard as Code

```json
{
  "dashboard": {
    "title": "AGI Eval Overview",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      }
    ]
  }
}
```

## Performance Monitoring

### SLA Monitoring

Key metrics to track:
- **Availability**: 99.9% uptime target
- **Response Time**: 95th percentile < 500ms
- **Throughput**: Handle 1000 concurrent evaluations
- **Error Rate**: < 0.1% of requests

### Performance Baselines

```python
# Define performance baselines
PERFORMANCE_BASELINES = {
    "api_response_time_p95": 0.5,  # 500ms
    "evaluation_duration_p95": 300,  # 5 minutes
    "database_query_time_p95": 0.1,  # 100ms
    "memory_usage_threshold": 0.8,  # 80%
    "cpu_usage_threshold": 0.7,  # 70%
}
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
docker stats
# Analyze heap dump
kubectl exec -it pod-name -- python -m memory_profiler script.py
```

#### Slow Database Queries
```sql
-- Find slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;
```

#### Failed Evaluations
```bash
# Check logs for specific evaluation
kubectl logs -l app=agi-eval --since=1h | grep "evaluation_id=eval-123"

# Check traces in Jaeger
curl "http://jaeger:16686/api/traces?service=agi-eval&tag=evaluation_id:eval-123"
```

### Debugging Workflows

1. **Start with metrics** - Check dashboards for anomalies
2. **Analyze logs** - Search for error patterns
3. **Trace requests** - Follow request flow through services
4. **Check external dependencies** - Verify third-party services
5. **Review recent changes** - Check deployment history

### Performance Optimization

#### Database Optimization
```sql
-- Add indexes for common queries
CREATE INDEX CONCURRENTLY idx_evaluations_status_created 
ON evaluations(status, created_at);

-- Analyze query plans
EXPLAIN ANALYZE SELECT * FROM evaluations WHERE status = 'running';
```

#### Caching Strategy
```python
# Redis caching for model responses
@cache(expire=3600)  # 1 hour cache
async def get_model_response(prompt: str, model: str) -> str:
    return await model_provider.generate(prompt, model)
```

#### Connection Pooling
```python
# Database connection pooling
DATABASE_POOL_SIZE = 20
DATABASE_MAX_OVERFLOW = 10

engine = create_async_engine(
    DATABASE_URL,
    pool_size=DATABASE_POOL_SIZE,
    max_overflow=DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True
)
```

## Security Monitoring

### Security Metrics
- Failed authentication attempts
- Suspicious API usage patterns
- Rate limiting triggers
- Access to sensitive endpoints

### Audit Logging
```python
@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    
    # Log security-relevant events
    if request.url.path.startswith("/admin"):
        audit_logger.info(
            "Admin access",
            extra={
                "user_id": request.state.user_id,
                "path": request.url.path,
                "method": request.method,
                "status_code": response.status_code,
                "duration": time.time() - start_time
            }
        )
    
    return response
```

## Cost Monitoring

### Resource Usage Tracking
- Cloud infrastructure costs
- Model API usage and costs
- Storage costs
- Data transfer costs

### Cost Alerts
```yaml
- alert: HighCloudCosts
  expr: increase(cloud_cost_total[24h]) > 100
  for: 1h
  labels:
    severity: warning
  annotations:
    summary: "Daily cloud costs exceeding budget"
    description: "Cloud costs are ${{ $value }} in the last 24 hours"
```

This comprehensive observability setup ensures full visibility into the AGI Evaluation Sandbox's performance, health, and business metrics.