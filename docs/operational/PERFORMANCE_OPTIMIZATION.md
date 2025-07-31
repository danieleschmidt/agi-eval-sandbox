# Performance Optimization Guide

## Overview

This document outlines performance optimization strategies for the AGI Evaluation Sandbox, covering application performance, infrastructure optimization, and monitoring best practices.

## Application Performance

### Python Optimization

#### Async/Await Best Practices
```python
# Optimized async patterns
import asyncio
import aiohttp
from typing import List, Dict, Any

class OptimizedEvaluator:
    """Performance-optimized evaluation engine"""
    
    def __init__(self):
        self.session = aiohttp.ClientSession(
            connector=aiohttp.TCPConnector(
                limit=100,  # Connection pool size
                limit_per_host=20,
                ttl_dns_cache=300,
                use_dns_cache=True,
            ),
            timeout=aiohttp.ClientTimeout(total=30)
        )
    
    async def evaluate_batch(self, prompts: List[str]) -> List[Dict[str, Any]]:
        """Evaluate prompts in parallel with optimized batching"""
        # Batch prompts for optimal throughput
        batch_size = 10
        batches = [prompts[i:i + batch_size] 
                  for i in range(0, len(prompts), batch_size)]
        
        # Process batches concurrently
        tasks = [self._process_batch(batch) for batch in batches]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results and handle exceptions
        flattened = []
        for result in results:
            if isinstance(result, Exception):
                # Log error and continue
                continue
            flattened.extend(result)
            
        return flattened
    
    async def _process_batch(self, batch: List[str]) -> List[Dict[str, Any]]:
        """Process a single batch with retry logic"""
        tasks = [self._evaluate_single(prompt) for prompt in batch]
        return await asyncio.gather(*tasks)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def _evaluate_single(self, prompt: str) -> Dict[str, Any]:
        """Evaluate single prompt with exponential backoff"""
        async with self.session.post(
            "https://api.openai.com/v1/chat/completions",
            json={"model": "gpt-4", "messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": f"Bearer {self.api_key}"}
        ) as response:
            return await response.json()
```

#### Memory Optimization
```python
# Memory-efficient data processing
import gc
from typing import Iterator, Generator
import psutil

class MemoryOptimizedProcessor:
    """Memory-conscious data processor"""
    
    def __init__(self, memory_threshold: float = 0.8):
        self.memory_threshold = memory_threshold
    
    def process_large_dataset(self, data_path: str) -> Generator[Dict, None, None]:
        """Process large datasets in chunks to avoid memory issues"""
        chunk_size = self._calculate_optimal_chunk_size()
        
        with open(data_path, 'r') as file:
            chunk = []
            for line in file:
                chunk.append(json.loads(line))
                
                if len(chunk) >= chunk_size:
                    yield from self._process_chunk(chunk)
                    chunk.clear()
                    
                    # Force garbage collection if memory usage is high
                    if self._get_memory_usage() > self.memory_threshold:
                        gc.collect()
            
            # Process remaining items
            if chunk:
                yield from self._process_chunk(chunk)
    
    def _calculate_optimal_chunk_size(self) -> int:
        """Calculate optimal chunk size based on available memory"""
        available_memory = psutil.virtual_memory().available
        estimated_item_size = 1024  # bytes per item estimate
        return min(10000, available_memory // (estimated_item_size * 10))
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        return psutil.virtual_memory().percent / 100.0
    
    def _process_chunk(self, chunk: List[Dict]) -> Iterator[Dict]:
        """Process a chunk of data"""
        for item in chunk:
            # Process item
            yield self._transform_item(item)
```

### Database Optimization

#### Query Optimization
```python
# Optimized database queries
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import QueuePool

class OptimizedDatabase:
    """Performance-optimized database operations"""
    
    def __init__(self, database_url: str):
        self.engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=30,
            pool_pre_ping=True,
            pool_recycle=3600,  # Recycle connections every hour
            echo=False  # Disable SQL logging in production
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
    
    async def get_evaluation_results(self, model_id: int, limit: int = 1000) -> List[Dict]:
        """Optimized query with proper indexing and pagination"""
        query = text(\"\"\"
            SELECT 
                er.id,
                er.benchmark_name,
                er.score,
                er.created_at,
                m.name as model_name
            FROM evaluation_results er
            JOIN models m ON er.model_id = m.id
            WHERE er.model_id = :model_id
            ORDER BY er.created_at DESC
            LIMIT :limit
        \"\"\")
        
        with self.engine.connect() as conn:
            result = conn.execute(query, {"model_id": model_id, "limit": limit})
            return [dict(row) for row in result]
    
    async def bulk_insert_results(self, results: List[Dict]) -> None:
        """Optimized bulk insert with batch processing"""
        batch_size = 1000
        
        with self.engine.begin() as conn:
            for i in range(0, len(results), batch_size):
                batch = results[i:i + batch_size]
                conn.execute(
                    text(\"\"\"
                        INSERT INTO evaluation_results 
                        (model_id, benchmark_name, score, metadata, created_at)
                        VALUES (:model_id, :benchmark_name, :score, :metadata, :created_at)
                    \"\"\"),
                    batch
                )
```

#### Connection Pooling
```yaml
# database-config.yml
database:
  connection_pool:
    min_size: 10
    max_size: 50
    max_overflow: 20
    pool_timeout: 30
    pool_recycle: 3600
    
  query_optimization:
    statement_timeout: "30s"
    lock_timeout: "10s"
    idle_in_transaction_session_timeout: "60s"
    
  indexing_strategy:
    - table: "evaluation_results"
      indexes:
        - columns: ["model_id", "created_at"]
          type: "btree"
        - columns: ["benchmark_name", "score"]
          type: "btree"
        - columns: ["metadata"]
          type: "gin"  # For JSON queries
```

### Caching Strategy

#### Redis Caching
```python
# Advanced caching implementation
import redis.asyncio as redis
import pickle
import hashlib
from typing import Optional, Any, Callable
import functools

class AdvancedCache:
    """Advanced caching with TTL and invalidation"""
    
    def __init__(self, redis_url: str):
        self.redis = redis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=False,  # Handle binary data
            max_connections=20
        )
    
    def cache_with_ttl(self, ttl: int = 3600, key_prefix: str = ""):
        """Decorator for caching function results with TTL"""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                # Generate cache key
                cache_key = self._generate_cache_key(func, args, kwargs, key_prefix)
                
                # Try to get from cache
                cached_result = await self.redis.get(cache_key)
                if cached_result:
                    return pickle.loads(cached_result)
                
                # Execute function and cache result
                result = await func(*args, **kwargs)
                await self.redis.setex(
                    cache_key, 
                    ttl, 
                    pickle.dumps(result, protocol=pickle.HIGHEST_PROTOCOL)
                )
                
                return result
            return wrapper
        return decorator
    
    def _generate_cache_key(self, func: Callable, args: tuple, kwargs: dict, prefix: str) -> str:
        """Generate deterministic cache key"""
        key_data = f"{prefix}:{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    async def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate all keys matching pattern"""
        keys = await self.redis.keys(pattern)
        if keys:
            return await self.redis.delete(*keys)
        return 0

# Usage example
cache = AdvancedCache("redis://localhost:6379")

@cache.cache_with_ttl(ttl=1800, key_prefix="eval")
async def get_model_evaluation(model_id: int, benchmark: str) -> Dict:
    """Cached model evaluation retrieval"""
    # Expensive database query
    return await database.get_evaluation_results(model_id, benchmark)
```

#### Application-Level Caching
```python
# In-memory caching for frequently accessed data
from cachetools import TTLCache, LRUCache
import threading

class ApplicationCache:
    """Thread-safe application-level caching"""
    
    def __init__(self):
        # TTL cache for temporary data (15 minutes)
        self.ttl_cache = TTLCache(maxsize=1000, ttl=900)
        
        # LRU cache for model metadata (persistent during runtime)
        self.lru_cache = LRUCache(maxsize=500)
        
        # Thread locks for cache access
        self.ttl_lock = threading.Lock()
        self.lru_lock = threading.Lock()
    
    def get_ttl(self, key: str) -> Optional[Any]:
        """Get from TTL cache with thread safety"""
        with self.ttl_lock:
            return self.ttl_cache.get(key)
    
    def set_ttl(self, key: str, value: Any) -> None:
        """Set in TTL cache with thread safety"""
        with self.ttl_lock:
            self.ttl_cache[key] = value
    
    def get_lru(self, key: str) -> Optional[Any]:
        """Get from LRU cache with thread safety"""
        with self.lru_lock:
            return self.lru_cache.get(key)
    
    def set_lru(self, key: str, value: Any) -> None:
        """Set in LRU cache with thread safety"""
        with self.lru_lock:
            self.lru_cache[key] = value
```

## Infrastructure Optimization

### Container Optimization

#### Multi-stage Dockerfile
```dockerfile
# Optimized multi-stage Dockerfile
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim as production

# Copy only necessary files from builder
COPY --from=builder /root/.local /root/.local

# Make sure scripts in .local are usable:
ENV PATH=/root/.local/bin:$PATH

# Add non-root user for security
RUN groupadd -r appuser && useradd -r -g appuser appuser

# Copy application code
COPY --chown=appuser:appuser . /app
WORKDIR /app

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Optimized startup
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--worker-connections", "1000", "main:app"]
```

#### Resource Optimization
```yaml
# k8s-resources.yml
resources:
  requests:
    cpu: "500m"
    memory: "1Gi"
  limits:
    cpu: "2000m"
    memory: "4Gi"

# Horizontal Pod Autoscaler
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: agi-eval-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: agi-eval-sandbox
  minReplicas: 3
  maxReplicas: 10
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
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
```

### Load Balancing and Traffic Management

#### Nginx Configuration
```nginx
# nginx-optimized.conf
upstream agi_eval_backend {
    # Least connections load balancing
    least_conn;
    
    server agi-eval-1:8000 max_fails=3 fail_timeout=30s;
    server agi-eval-2:8000 max_fails=3 fail_timeout=30s;
    server agi-eval-3:8000 max_fails=3 fail_timeout=30s;
    
    # Keep alive connections
    keepalive 32;
}

server {
    listen 80;
    server_name api.agi-eval.your-org.com;
    
    # Redirect HTTP to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.agi-eval.your-org.com;
    
    # SSL configuration
    ssl_certificate /etc/ssl/certs/agi-eval.crt;
    ssl_certificate_key /etc/ssl/private/agi-eval.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512:ECDHE-RSA-AES256-GCM-SHA384;
    ssl_prefer_server_ciphers off;
    
    # Performance optimizations
    gzip on;
    gzip_vary on;
    gzip_min_length 1024;
    gzip_types
        text/plain
        text/css
        text/xml
        text/javascript
        application/json
        application/javascript
        application/xml+rss
        application/atom+xml;
    
    # Client body optimization
    client_max_body_size 10M;
    client_body_buffer_size 128k;
    client_body_timeout 12;
    client_header_timeout 12;
    
    # Connection keep alive
    keepalive_timeout 65;
    keepalive_requests 100;
    
    location / {
        proxy_pass http://agi_eval_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        
        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Buffer optimization
        proxy_buffering on;
        proxy_buffer_size 4k;
        proxy_buffers 8 4k;
        proxy_busy_buffers_size 8k;
    }
    
    # Static file caching
    location ~* \.(jpg|jpeg|png|gif|ico|css|js)$ {
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
    
    # API response caching for stable endpoints
    location ~* ^/api/v1/(models|benchmarks)$ {
        proxy_pass http://agi_eval_backend;
        proxy_cache api_cache;
        proxy_cache_valid 200 5m;
        proxy_cache_use_stale error timeout invalid_header updating;
        add_header X-Cache-Status $upstream_cache_status;
    }
}

# Cache configuration
proxy_cache_path /var/cache/nginx/api levels=1:2 keys_zone=api_cache:10m max_size=100m inactive=60m use_temp_path=off;
```

#### Service Mesh (Istio) Configuration
```yaml
# istio-performance.yml
apiVersion: networking.istio.io/v1alpha3
kind: DestinationRule
metadata:
  name: agi-eval-destination
spec:
  host: agi-eval-service
  trafficPolicy:
    loadBalancer:
      simple: LEAST_CONN
    connectionPool:
      tcp:
        maxConnections: 100
        connectTimeout: 30s
        keepAlive:
          time: 7200s
          interval: 75s
      http:
        http1MaxPendingRequests: 64
        http2MaxRequests: 100
        maxRequestsPerConnection: 2
        maxRetries: 3
        consecutiveGatewayErrors: 5
        interval: 30s
        baseEjectionTime: 30s
        
---
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: agi-eval-virtual-service
spec:
  hosts:
    - api.agi-eval.your-org.com
  http:
    - match:
        - uri:
            prefix: /api/v1/evaluate
      route:
        - destination:
            host: agi-eval-service
      timeout: 300s  # Long timeout for evaluation requests
      retries:
        attempts: 3
        perTryTimeout: 100s
        
    - match:
        - uri:
            prefix: /
      route:
        - destination:
            host: agi-eval-service
      timeout: 30s
      retries:
        attempts: 2
        perTryTimeout: 10s
```

## Monitoring and Observability

### Performance Metrics

#### Custom Metrics Collection
```python
# Performance monitoring
from prometheus_client import Counter, Histogram, Gauge, start_http_server
import time
import functools

# Metrics definitions
REQUEST_COUNT = Counter('api_requests_total', 'Total API requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('api_request_duration_seconds', 'API request duration', ['method', 'endpoint'])
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active connections')
EVALUATION_QUEUE_SIZE = Gauge('evaluation_queue_size', 'Evaluation queue size')
MODEL_RESPONSE_TIME = Histogram('model_response_time_seconds', 'Model API response time', ['model', 'provider'])

def monitor_performance(func):
    """Decorator to monitor function performance"""
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        start_time = time.time()
        
        try:
            result = await func(*args, **kwargs)
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status='success'
            ).inc()
            return result
            
        except Exception as e:
            REQUEST_COUNT.labels(
                method=request.method,
                endpoint=request.url.path,
                status='error'
            ).inc()
            raise
            
        finally:
            REQUEST_DURATION.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(time.time() - start_time)
            
    return wrapper

class PerformanceMonitor:
    """Advanced performance monitoring"""
    
    def __init__(self):
        self.start_time = time.time()
        
    async def collect_system_metrics(self):
        """Collect system-level performance metrics"""
        import psutil
        
        # CPU and memory metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Custom metrics
        SYSTEM_CPU_USAGE.set(cpu_percent)
        SYSTEM_MEMORY_USAGE.set(memory.percent)
        SYSTEM_DISK_USAGE.set(disk.percent)
        
        # Application metrics
        ACTIVE_CONNECTIONS.set(len(self.get_active_connections()))
        EVALUATION_QUEUE_SIZE.set(await self.get_queue_size())
    
    async def monitor_model_performance(self, model: str, provider: str, response_time: float):
        """Monitor model API performance"""
        MODEL_RESPONSE_TIME.labels(model=model, provider=provider).observe(response_time)
```

#### Grafana Dashboard Configuration
```json
{
  "dashboard": {
    "title": "AGI Eval Sandbox Performance",
    "panels": [
      {
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(api_requests_total[5m])",
            "legendFormat": "{{method}} {{endpoint}}"
          }
        ]
      },
      {
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          },
          {
            "expr": "histogram_quantile(0.50, rate(api_request_duration_seconds_bucket[5m]))",
            "legendFormat": "50th percentile"
          }
        ]
      },
      {
        "title": "Error Rate",
        "type": "singlestat",
        "targets": [
          {
            "expr": "rate(api_requests_total{status=\"error\"}[5m]) / rate(api_requests_total[5m]) * 100"
          }
        ]
      },
      {
        "title": "System Resources",
        "type": "graph",
        "targets": [
          {
            "expr": "system_cpu_usage",
            "legendFormat": "CPU %"
          },
          {
            "expr": "system_memory_usage",
            "legendFormat": "Memory %"
          }
        ]
      }
    ]
  }
}
```

### Profiling and Debugging

#### Application Profiling
```python
# Performance profiling utilities
import cProfile
import pstats
import io
from contextlib import contextmanager

@contextmanager
def profile_code():
    """Context manager for profiling code blocks"""
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        yield profiler
    finally:
        profiler.disable()
        
        # Generate report
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats()
        
        # Log profile results
        logger.info(f"Profile results:\n{s.getvalue()}")

# Usage example
async def expensive_operation():
    with profile_code():
        # Expensive computation here
        result = await complex_evaluation()
    return result
```

#### Memory Profiling
```python
# Memory usage monitoring
import psutil
import tracemalloc
from typing import Dict, Any

class MemoryProfiler:
    """Memory usage profiler"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
        if enabled:
            tracemalloc.start()
    
    def get_memory_snapshot(self) -> Dict[str, Any]:
        """Get current memory usage snapshot"""
        if not self.enabled:
            return {}
            
        # Process memory info
        process = psutil.Process()
        memory_info = process.memory_info()
        
        # Python memory tracing
        current, peak = tracemalloc.get_traced_memory()
        
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
            "python_current_mb": current / 1024 / 1024,
            "python_peak_mb": peak / 1024 / 1024,
            "memory_percent": process.memory_percent()
        }
    
    def log_top_memory_usage(self, limit: int = 10):
        """Log top memory usage by file/line"""
        if not self.enabled:
            return
            
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        logger.info(f"Top {limit} memory usage:")
        for stat in top_stats[:limit]:
            logger.info(f"{stat}")
```

## Performance Testing

### Load Testing Configuration

#### Artillery Configuration
```yaml
# artillery-config.yml
config:
  target: 'https://api.agi-eval.your-org.com'
  phases:
    # Warm-up phase
    - duration: 60
      arrivalRate: 5
      name: "Warm-up"
      
    # Ramp-up phase
    - duration: 300
      arrivalRate: 10
      rampTo: 50
      name: "Ramp-up"
      
    # Sustained load
    - duration: 600
      arrivalRate: 50
      name: "Sustained load"
      
    # Peak load
    - duration: 180
      arrivalRate: 100
      name: "Peak load"
      
    # Cool-down
    - duration: 60
      arrivalRate: 10
      name: "Cool-down"
      
  defaults:
    headers:
      Authorization: "Bearer {{$randomString()}}"
      Content-Type: "application/json"

scenarios:
  - name: "Health check"
    weight: 10
    flow:
      - get:
          url: "/health"
          
  - name: "List models"
    weight: 20
    flow:
      - get:
          url: "/api/v1/models"
          
  - name: "Model evaluation"
    weight: 70
    flow:
      - post:
          url: "/api/v1/evaluate"
          json:
            model: "gpt-4"
            benchmark: "mmlu"
            prompts: ["{{ $randomString() }}"]
          capture:
            - json: "$.job_id"
              as: "job_id"
      - get:
          url: "/api/v1/jobs/{{ job_id }}"
```

#### K6 Performance Testing
```javascript
// k6-performance-test.js
import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

// Custom metrics
const errorRate = new Rate('errors');
const responseTime = new Trend('response_time');

export const options = {
  stages: [
    { duration: '2m', target: 10 },   // Ramp-up
    { duration: '5m', target: 50 },   // Stay at 50 users
    { duration: '2m', target: 100 },  // Ramp-up to 100 users
    { duration: '5m', target: 100 },  // Stay at 100 users
    { duration: '2m', target: 0 },    // Ramp-down
  ],
  thresholds: {
    http_req_duration: ['p(95)<500'], // 95% of requests under 500ms
    http_req_failed: ['rate<0.1'],    // Error rate under 10%
    errors: ['rate<0.1'],
  },
};

export default function () {
  const baseURL = 'https://api.agi-eval.your-org.com';
  
  // Test different endpoints
  const responses = http.batch([
    ['GET', `${baseURL}/health`],
    ['GET', `${baseURL}/api/v1/models`],
    ['POST', `${baseURL}/api/v1/evaluate`, JSON.stringify({
      model: 'gpt-4',
      benchmark: 'mmlu',
      prompts: ['What is 2+2?']
    }), { headers: { 'Content-Type': 'application/json' } }],
  ]);
  
  responses.forEach((response, index) => {
    const success = check(response, {
      'status is 200': (r) => r.status === 200,
      'response time OK': (r) => r.timings.duration < 500,
    });
    
    errorRate.add(!success);
    responseTime.add(response.timings.duration);
  });
  
  sleep(1);
}
```

## Performance Optimization Checklist

### Application Level
- [ ] Implement async/await patterns correctly
- [ ] Use connection pooling for databases
- [ ] Implement multi-level caching strategy
- [ ] Optimize database queries and indexes
- [ ] Use batch processing for bulk operations
- [ ] Implement circuit breakers for external APIs
- [ ] Profile and optimize hot code paths

### Infrastructure Level
- [ ] Configure horizontal pod autoscaling
- [ ] Optimize container resource limits
- [ ] Implement efficient load balancing
- [ ] Use CDN for static content
- [ ] Configure connection keep-alive
- [ ] Optimize network policies
- [ ] Set up monitoring and alerting

### Database Level
- [ ] Create appropriate indexes
- [ ] Optimize query performance
- [ ] Configure connection pooling
- [ ] Implement read replicas
- [ ] Set up query caching
- [ ] Monitor slow queries
- [ ] Regular maintenance tasks

### Monitoring Level
- [ ] Set up performance metrics collection
- [ ] Configure alerting thresholds
- [ ] Implement distributed tracing
- [ ] Monitor resource utilization
- [ ] Track user experience metrics
- [ ] Set up performance dashboards
- [ ] Regular performance reviews

## Related Documentation

- [Monitoring and Observability](../operational/MONITORING.md)
- [Infrastructure Requirements](../operational/INFRASTRUCTURE.md)
- [Load Testing Guide](../testing/LOAD_TESTING.md)
- [Troubleshooting Guide](../operational/TROUBLESHOOTING.md)

---

**Note**: Performance optimization is an iterative process. Regular monitoring, profiling, and testing are essential to maintain optimal performance as the application scales.