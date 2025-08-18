# Operational Runbooks

This directory contains operational runbooks for common scenarios and incident response procedures for the AGI Evaluation Sandbox.

## Directory Structure

```
runbooks/
├── README.md                    # This file
├── incidents/                  # Incident response runbooks
│   ├── service-outage.md       # Service outage response
│   ├── performance-degradation.md # Performance issue response
│   ├── security-incident.md    # Security incident response
│   └── data-corruption.md      # Data corruption response
├── maintenance/                # Maintenance procedures
│   ├── database-maintenance.md # Database maintenance tasks
│   ├── system-updates.md      # System update procedures
│   ├── backup-recovery.md     # Backup and recovery procedures
│   └── scaling-operations.md  # Scaling procedures
├── monitoring/                 # Monitoring and alerting
│   ├── alert-response.md      # Alert response procedures
│   ├── dashboard-guide.md     # Monitoring dashboard guide
│   └── health-checks.md       # Health check procedures
└── troubleshooting/           # Troubleshooting guides
    ├── common-issues.md       # Common issues and solutions
    ├── performance-tuning.md  # Performance optimization
    └── debugging-guide.md     # Debugging procedures
```

## Quick Reference

### Emergency Contacts
- **On-call Engineer:** Check PagerDuty rotation
- **Platform Team:** #platform-alerts Slack channel
- **Security Team:** #security-incidents Slack channel

### Critical System URLs
- **Production Dashboard:** https://dashboard.agi-eval.example.com
- **Monitoring:** https://monitoring.agi-eval.example.com
- **Grafana:** https://grafana.agi-eval.example.com
- **Status Page:** https://status.agi-eval.example.com

### Quick Actions

#### Check System Health
```bash
# Check API health
curl -f https://api.agi-eval.example.com/health

# Check database connection
docker exec agi-eval-db pg_isready

# Check Redis
docker exec agi-eval-redis redis-cli ping
```

#### View Recent Logs
```bash
# API logs
kubectl logs -n agi-eval deployment/api --tail=100

# Worker logs
kubectl logs -n agi-eval deployment/worker --tail=100

# Database logs
kubectl logs -n agi-eval statefulset/database --tail=100
```

#### Scale Services
```bash
# Scale API replicas
kubectl scale deployment/api --replicas=5

# Scale worker replicas
kubectl scale deployment/worker --replicas=10
```

## Runbook Usage Guidelines

1. **Follow the Checklist:** Each runbook contains step-by-step checklists
2. **Document Actions:** Record all actions taken during incidents
3. **Update Runbooks:** Keep runbooks current with system changes
4. **Test Procedures:** Regularly test runbook procedures during maintenance windows
5. **Post-Incident Reviews:** Update runbooks based on lessons learned

## Escalation Matrix

| Severity | Response Time | Escalation Path |
|----------|---------------|-----------------|
| P0 (Critical) | 15 minutes | On-call → Team Lead → Engineering Manager |
| P1 (High) | 1 hour | On-call → Team Lead |
| P2 (Medium) | 4 hours | On-call Engineer |
| P3 (Low) | Next business day | Team backlog |

## Communication Channels

### Internal
- **#platform-alerts:** Automated alerts and status updates
- **#incidents:** Active incident coordination
- **#platform-team:** Team discussions and updates
- **#engineering:** Broader engineering notifications

### External
- **Status Page:** Public status updates
- **Support Portal:** Customer communication
- **Email Notifications:** Critical customer notifications

## Metrics and SLIs

### Service Level Indicators (SLIs)
- **Availability:** 99.9% uptime target
- **Latency:** 95th percentile < 2 seconds
- **Error Rate:** < 0.1% error rate
- **Throughput:** Support peak load of 1000 RPS

### Key Metrics to Monitor
- Response times across all endpoints
- Error rates by service and endpoint
- CPU and memory utilization
- Database connection pool status
- Queue depth and processing times
- Active user sessions

## Security Considerations

- **Access Control:** All runbook procedures require appropriate access levels
- **Audit Logging:** All operational actions are logged and auditable
- **Sensitive Data:** Never log or expose sensitive data during troubleshooting
- **Incident Response:** Follow security incident response procedures for security-related issues

---

**Remember:** When in doubt, escalate early and communicate frequently. It's better to over-communicate during incidents than to work in isolation.