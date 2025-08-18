# Service Outage Response Runbook

## Overview
This runbook provides step-by-step procedures for responding to complete or partial service outages.

## Severity: P0 (Critical)
**Target Response Time:** 15 minutes  
**Target Resolution Time:** 2 hours

## Detection
Service outages are typically detected through:
- Automated monitoring alerts (Prometheus/Grafana)
- Health check failures
- Customer reports
- External monitoring services
- Load balancer health checks

## Immediate Response (0-15 minutes)

### 1. Acknowledge and Assess
- [ ] Acknowledge the incident in PagerDuty
- [ ] Join the incident response channel: `#incidents`
- [ ] Post initial status: "Investigating service outage - [timestamp]"
- [ ] Check status page and update if necessary

### 2. Quick Health Assessment
```bash
# Check API health endpoint
curl -f https://api.agi-eval.example.com/health

# Check main services status
kubectl get pods -n agi-eval

# Check service endpoints
curl -f https://api.agi-eval.example.com/api/v1/status
```

### 3. Determine Impact Scope
- [ ] Identify affected services (API, Dashboard, Workers)
- [ ] Check geographic impact (multi-region deployments)
- [ ] Assess customer impact level
- [ ] Update incident severity if needed

## Investigation Phase (15-60 minutes)

### 4. System Status Check
```bash
# Check cluster health
kubectl cluster-info

# Check node status
kubectl get nodes

# Check resource utilization
kubectl top nodes
kubectl top pods -n agi-eval
```

### 5. Service-Specific Diagnostics

#### API Service Issues
```bash
# Check API pods
kubectl get pods -n agi-eval -l app=api

# View recent API logs
kubectl logs -n agi-eval deployment/api --tail=500

# Check API metrics
curl -s http://prometheus:9090/api/v1/query?query=up{job="api"}
```

#### Database Issues
```bash
# Check database connectivity
kubectl exec -it deployment/api -- pg_isready -h database

# Check database status
kubectl logs -n agi-eval statefulset/database --tail=200

# Monitor database metrics
curl -s http://prometheus:9090/api/v1/query?query=pg_up
```

#### Redis Issues
```bash
# Check Redis connectivity
kubectl exec -it deployment/api -- redis-cli -h redis ping

# Check Redis memory usage
kubectl exec -it deployment/redis -- redis-cli info memory
```

### 6. Infrastructure Assessment
- [ ] Check cloud provider status pages
- [ ] Verify DNS resolution
- [ ] Check CDN status
- [ ] Verify load balancer configuration
- [ ] Check SSL certificate validity

## Common Resolution Steps

### 7. Service Restart
```bash
# Restart API deployment
kubectl rollout restart deployment/api -n agi-eval

# Restart worker deployment
kubectl rollout restart deployment/worker -n agi-eval

# Wait for rollout to complete
kubectl rollout status deployment/api -n agi-eval
```

### 8. Configuration Issues
```bash
# Check ConfigMaps
kubectl get configmaps -n agi-eval

# Check Secrets
kubectl get secrets -n agi-eval

# Validate environment variables
kubectl exec -it deployment/api -- env | grep -E "(DATABASE|REDIS|API)"
```

### 9. Resource Scaling
```bash
# Scale up if resource constrained
kubectl scale deployment/api --replicas=5 -n agi-eval
kubectl scale deployment/worker --replicas=10 -n agi-eval

# Check pod distribution
kubectl get pods -n agi-eval -o wide
```

## Recovery Verification

### 10. Service Validation
```bash
# Test API functionality
curl -f https://api.agi-eval.example.com/api/v1/health
curl -f https://api.agi-eval.example.com/api/v1/benchmarks

# Test evaluation workflow
# (Use internal test scripts)
./scripts/smoke-test.sh

# Verify dashboard access
curl -f https://dashboard.agi-eval.example.com
```

### 11. Monitor Recovery Metrics
- [ ] Check response times return to baseline
- [ ] Verify error rates drop to normal levels
- [ ] Monitor resource utilization
- [ ] Check queue processing rates
- [ ] Validate all health checks pass

## Communication

### 12. Status Updates
- [ ] Update incident channel every 15 minutes during active response
- [ ] Update status page with current status
- [ ] Notify affected customers if impact is significant
- [ ] Send recovery notification once service is restored

### Sample Communication Templates

#### Initial Alert
```
🚨 INCIDENT: Service outage detected
Time: [timestamp]
Impact: [description of impact]
Status: Investigating
ETA: Updates every 15 minutes
```

#### Progress Update
```
📊 UPDATE: Service outage investigation
Time: [timestamp]
Progress: [current actions being taken]
Next Update: [timestamp + 15 minutes]
```

#### Resolution Notice
```
✅ RESOLVED: Service outage resolved
Time: [timestamp]
Duration: [total outage duration]
Root Cause: [brief description]
Follow-up: Post-incident review scheduled
```

## Post-Incident Activities

### 13. Service Stabilization
- [ ] Monitor service for 1 hour post-recovery
- [ ] Ensure all dependent services are functioning
- [ ] Verify no secondary issues have emerged
- [ ] Scale resources back to normal if temporarily increased

### 14. Documentation
- [ ] Record all actions taken in incident timeline
- [ ] Document root cause analysis
- [ ] Update monitoring alerts if gaps were identified
- [ ] Schedule post-incident review meeting

### 15. Follow-up Actions
- [ ] Create post-incident review document
- [ ] Identify preventive measures
- [ ] Update runbooks based on lessons learned
- [ ] Implement monitoring improvements
- [ ] Schedule any necessary infrastructure improvements

## Escalation Triggers

Escalate to Engineering Manager if:
- Resolution time exceeds 2 hours
- Multiple services are affected
- Data loss is suspected
- Security implications are identified
- Customer impact is severe

Escalate to CTO if:
- Resolution time exceeds 4 hours
- Business-critical functionality is offline
- Regulatory compliance is at risk
- Media attention is likely

## Prevention Measures

- Implement comprehensive monitoring and alerting
- Regular disaster recovery testing
- Automated health checks and failover
- Capacity planning and load testing
- Regular security updates and patches
- Infrastructure as Code for consistent deployments

## Related Runbooks
- [Performance Degradation Response](./performance-degradation.md)
- [Database Maintenance](../maintenance/database-maintenance.md)
- [Alert Response Procedures](../monitoring/alert-response.md)

---
**Last Updated:** [Current Date]  
**Review Frequency:** Monthly  
**Owner:** Platform Team