# ADR-003: Celery with Redis for Distributed Task Processing

## Status
Accepted

## Context
We need a distributed task processing system for:
- Long-running evaluation jobs that can take hours
- Parallel execution of benchmark tasks across multiple workers
- Retry logic and failure handling for model API calls
- Monitoring and observability of task progress
- Horizontal scaling of evaluation capacity

## Decision
We will use Celery with Redis as the message broker for distributed task processing.

## Consequences

### Positive
- Battle-tested solution with extensive production usage
- Excellent monitoring and debugging tools (Flower, etc.)
- Flexible task routing and priority management
- Built-in retry logic and error handling
- Horizontal scaling by adding worker nodes
- Redis provides both message broker and result backend

### Negative
- Additional infrastructure complexity (Redis cluster for production)
- Memory usage can be high for large result payloads
- Potential for task state inconsistencies during failures

## Alternatives Considered

### Apache Airflow
- **Pros**: Excellent workflow orchestration, rich UI, complex dependency handling
- **Cons**: Heavy for simple task queuing, complex setup, designed for batch processing

### RQ (Redis Queue)
- **Pros**: Simpler than Celery, Python-native, good for basic use cases
- **Cons**: Limited features, no multi-broker support, smaller ecosystem

### AWS SQS + Lambda
- **Pros**: Serverless, managed service, automatic scaling
- **Cons**: Vendor lock-in, cold start latency, limited execution time

### Kubernetes Jobs
- **Pros**: Native container orchestration, resource management
- **Cons**: Less flexible task management, complex for dynamic workloads

## Implementation Notes
- Use Redis Cluster for production high availability
- Implement custom task serialization for large evaluation payloads
- Configure appropriate concurrency limits per worker type
- Use task routing to dedicate workers for specific model types
- Implement comprehensive task monitoring and alerting

## References
- [Celery Documentation](https://docs.celeryproject.org/en/stable/)
- [Redis Documentation](https://redis.io/documentation)
- [Celery Best Practices](https://docs.celeryproject.org/en/stable/userguide/tasks.html#best-practices)