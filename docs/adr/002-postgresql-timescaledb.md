# ADR-002: PostgreSQL with TimescaleDB for Time-Series Data

## Status
Accepted

## Context
We need a database solution that can handle:
- Relational data (models, benchmarks, users)
- Time-series data (evaluation results over time)
- Complex queries for analytics and reporting
- ACID compliance for data integrity
- Horizontal scaling capabilities

## Decision
We will use PostgreSQL 15+ as our primary database with TimescaleDB extension for time-series workloads.

## Consequences
**Positive:**
- ACID compliance ensures data integrity
- Rich SQL feature set for complex analytics
- TimescaleDB provides optimized time-series storage and queries
- Excellent Python ecosystem support (SQLAlchemy, asyncpg)
- Battle-tested in production environments
- JSON/JSONB support for flexible schema evolution

**Negative:**
- More complex to operate than NoSQL solutions
- Vertical scaling limitations require sharding for massive scale
- TimescaleDB adds operational complexity

## Alternatives Considered
- **MongoDB**: Lacks ACID guarantees, less suitable for analytics
- **InfluxDB**: Great for time-series but poor for relational data
- **MySQL**: Less advanced JSON support and analytics features
- **Cassandra**: Complex operational overhead, eventual consistency

## Implementation Notes
- Use asyncpg for high-performance async database operations
- Implement proper indexing strategy for time-series queries
- Set up read replicas for analytics workloads
- Use connection pooling (pgbouncer) for production deployments
- Implement automated backup and point-in-time recovery

## References
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [PostgreSQL Performance Tuning](https://wiki.postgresql.org/wiki/Performance_Optimization)
