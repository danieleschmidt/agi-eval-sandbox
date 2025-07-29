# Disaster Recovery Plan

This document outlines the disaster recovery procedures for the AGI Evaluation Sandbox.

## Recovery Objectives

### Recovery Time Objective (RTO)
- **Critical Services**: 4 hours
- **Non-Critical Services**: 24 hours
- **Development Environment**: 48 hours

### Recovery Point Objective (RPO)
- **Primary Database**: 15 minutes
- **Configuration Data**: 1 hour
- **Code Repository**: Real-time (Git)
- **User Data**: 1 hour

## Disaster Scenarios

### Scenario 1: Complete Data Center Outage

**Impact**: Total service unavailability
**Probability**: Low
**RTO**: 4 hours
**RPO**: 15 minutes

#### Recovery Steps

1. **Assessment Phase (0-30 minutes)**
   ```bash
   # Verify outage scope
   curl -I https://api.agi-eval.com/health
   nslookup api.agi-eval.com
   
   # Check cloud provider status
   aws ec2 describe-instances --region us-east-1
   ```

2. **Activation Phase (30-60 minutes)**
   ```bash
   # Activate disaster recovery site
   terraform apply -var="environment=dr" -var="region=us-west-2"
   
   # Update DNS to point to DR site
   aws route53 change-resource-record-sets --hosted-zone-id Z123456789 \
     --change-batch file://dns-failover.json
   ```

3. **Recovery Phase (1-4 hours)**
   ```bash
   # Restore database from backup
   aws rds restore-db-instance-from-db-snapshot \
     --db-instance-identifier agi-eval-dr \
     --db-snapshot-identifier latest-snapshot
   
   # Deploy application
   kubectl apply -f k8s/disaster-recovery/
   
   # Verify services
   kubectl get pods -n agi-eval
   curl -f https://dr.agi-eval.com/health
   ```

### Scenario 2: Database Corruption

**Impact**: Data integrity issues
**Probability**: Medium
**RTO**: 2 hours
**RPO**: 15 minutes

#### Recovery Steps

```bash
# 1. Stop application to prevent further corruption
kubectl scale deployment api --replicas=0

# 2. Assess corruption extent
pg_dump --schema-only postgres://user:pass@host/db > schema_check.sql
pg_restore --list backup.dump | grep -i error

# 3. Restore from point-in-time backup
aws rds restore-db-instance-to-point-in-time \
  --source-db-instance-identifier agi-eval-prod \
  --target-db-instance-identifier agi-eval-restore \
  --restore-time 2024-01-01T12:00:00Z

# 4. Validate data integrity
python scripts/validate_data_integrity.py

# 5. Switch traffic to restored database
kubectl set env deployment/api DATABASE_URL=postgresql://restored-host/db

# 6. Scale application back up
kubectl scale deployment api --replicas=3
```

### Scenario 3: Security Breach

**Impact**: Potential data exposure
**Probability**: Medium
**RTO**: 1 hour (containment)
**RPO**: N/A

#### Recovery Steps

```bash
# 1. Immediate containment
kubectl delete ingress api-ingress  # Block external access
kubectl scale deployment api --replicas=0  # Stop services

# 2. Preserve evidence
kubectl logs deployment/api > incident-logs.txt
pg_dump postgres://host/db > forensic-backup.sql

# 3. Assess impact
python scripts/security_audit.py --breach-analysis
grep -r "suspicious_activity" /var/log/

# 4. Clean and rebuild
docker build -t agi-eval:clean --no-cache .
kubectl set image deployment/api api=agi-eval:clean

# 5. Implement additional security
kubectl apply -f k8s/security-hardening/
```

## Backup Strategy

### Database Backups

```yaml
# Automated backup schedule
apiVersion: batch/v1
kind: CronJob
metadata:
  name: db-backup
spec:
  schedule: "0 */6 * * *"  # Every 6 hours
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: backup
            image: postgres:15
            command:
            - /bin/bash
            - -c
            - |
              pg_dump $DATABASE_URL | gzip > /backup/db-$(date +%Y%m%d-%H%M%S).sql.gz
              aws s3 cp /backup/ s3://agi-eval-backups/database/ --recursive
```

### File System Backups

```bash
# Configuration backup
tar -czf config-backup-$(date +%Y%m%d).tar.gz \
  /etc/agi-eval/ \
  /var/lib/agi-eval/config/ \
  ~/.aws/

# Upload to S3
aws s3 cp config-backup-*.tar.gz s3://agi-eval-backups/config/
```

### Code Repository Backup

```bash
# Mirror repository to backup location
git clone --mirror https://github.com/your-org/agi-eval-sandbox.git
cd agi-eval-sandbox.git
git remote add backup https://backup-git-server/agi-eval-sandbox.git
git push backup --mirror
```

## Infrastructure as Code

### Terraform Disaster Recovery

```hcl
# dr-infrastructure.tf
module "disaster_recovery" {
  source = "./modules/infrastructure"
  
  environment = "dr"
  region      = "us-west-2"
  
  # Reduced capacity for cost optimization
  api_instance_count = 2
  db_instance_class  = "db.t3.medium"
  
  # Cross-region replication
  enable_cross_region_backup = true
  backup_retention_period    = 7
}
```

### Kubernetes Disaster Recovery

```yaml
# disaster-recovery-namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: agi-eval-dr
  labels:
    environment: disaster-recovery
---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-dr
  namespace: agi-eval-dr
spec:
  replicas: 2
  selector:
    matchLabels:
      app: api-dr
  template:
    spec:
      containers:
      - name: api
        image: agi-eval-sandbox:latest
        env:
        - name: DATABASE_URL
          value: "postgresql://dr-db:5432/agi_eval"
        - name: ENVIRONMENT
          value: "disaster-recovery"
```

## Communication Plan

### Stakeholder Notification

```python
# disaster_recovery_notifications.py
def notify_stakeholders(disaster_type, severity):
    recipients = {
        'executive': ['ceo@company.com', 'cto@company.com'],
        'technical': ['devops@company.com', 'engineering@company.com'],
        'customer': ['support@company.com'],
        'legal': ['legal@company.com']
    }
    
    message = f"""
    DISASTER RECOVERY ACTIVATED
    
    Type: {disaster_type}
    Severity: {severity}
    Time: {datetime.now()}
    
    Status: Recovery procedures initiated
    ETA: {calculate_eta(disaster_type)}
    
    Next Update: {datetime.now() + timedelta(hours=1)}
    """
    
    send_notifications(recipients, message)
```

### Status Page Updates

```bash
# Update status page
curl -X POST https://api.statuspage.io/v1/pages/PAGE_ID/incidents \
  -H "Authorization: OAuth TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "incident": {
      "name": "Disaster Recovery Activated",
      "status": "investigating",
      "impact_override": "major",
      "body": "We are currently recovering from a service disruption. All hands are on deck working to restore service."
    }
  }'
```

## Testing and Validation

### Monthly DR Tests

```bash
#!/bin/bash
# monthly_dr_test.sh

echo "Starting monthly disaster recovery test..."

# 1. Create test database snapshot
aws rds create-db-snapshot \
  --db-instance-identifier agi-eval-test \
  --db-snapshot-identifier dr-test-$(date +%Y%m%d)

# 2. Deploy to DR environment
terraform plan -var="environment=dr-test"
terraform apply -var="environment=dr-test" -auto-approve

# 3. Validate functionality
pytest tests/dr_validation/ -v

# 4. Measure RTO/RPO
python scripts/measure_recovery_metrics.py

# 5. Clean up test resources
terraform destroy -var="environment=dr-test" -auto-approve

echo "DR test completed. Results saved to dr-test-report-$(date +%Y%m%d).json"
```

### Automated Validation Scripts

```python
# dr_validation.py
import requests
import psycopg2
from datetime import datetime

def validate_api_endpoints():
    """Test critical API endpoints"""
    endpoints = [
        '/health',
        '/api/v1/evaluations',
        '/api/v1/models',
        '/api/v1/benchmarks'
    ]
    
    for endpoint in endpoints:
        response = requests.get(f"https://dr.agi-eval.com{endpoint}")
        assert response.status_code == 200, f"Endpoint {endpoint} failed"

def validate_database_integrity():
    """Check database consistency"""
    conn = psycopg2.connect(DATABASE_URL_DR)
    cursor = conn.cursor()
    
    # Check table counts
    cursor.execute("SELECT COUNT(*) FROM evaluations")
    eval_count = cursor.fetchone()[0]
    assert eval_count > 0, "No evaluations found in DR database"
    
    # Check data integrity
    cursor.execute("SELECT COUNT(*) FROM models WHERE created_at IS NULL")
    null_count = cursor.fetchone()[0]
    assert null_count == 0, "Data integrity issues found"

def validate_performance():
    """Check system performance meets SLA"""
    start_time = datetime.now()
    response = requests.get("https://dr.agi-eval.com/api/v1/health")
    response_time = (datetime.now() - start_time).total_seconds()
    
    assert response_time < 2.0, f"Response time {response_time}s exceeds SLA"
```

## Recovery Metrics and Reporting

### Metrics Collection

```yaml
# prometheus-dr-metrics.yml
groups:
  - name: disaster_recovery_metrics
    rules:
      - record: dr_test_success_rate
        expr: |
          (
            count(increase(dr_test_passed_total[30d])) /
            count(increase(dr_test_total[30d]))
          ) * 100
      
      - record: actual_rto_hours
        expr: |
          histogram_quantile(0.95, dr_recovery_time_hours_bucket)
      
      - record: actual_rpo_minutes
        expr: |
          histogram_quantile(0.95, dr_data_loss_minutes_bucket)
```

### Monthly DR Report

```python
# generate_dr_report.py
def generate_monthly_report():
    return {
        'period': '2024-01',
        'tests_conducted': 4,
        'tests_passed': 4,
        'average_rto': '2.5 hours',
        'average_rpo': '8 minutes',
        'issues_found': [
            'DNS propagation delay in us-west-2',
            'SSL certificate renewal needed for DR domain'
        ],
        'improvements_made': [
            'Automated SSL certificate management',
            'Improved monitoring for DR environment'
        ],
        'next_month_focus': [
            'Test cross-region database replication',
            'Validate backup restoration procedures'
        ]
    }
```

## Cost Optimization

### Hot-Standby vs Cold-Standby

```hcl
# Cost-optimized DR configuration
variable "dr_mode" {
  description = "DR mode: hot, warm, or cold"
  type        = string
  default     = "warm"
}

locals {
  instance_counts = {
    hot  = { api = 3, db = "db.r5.large" }
    warm = { api = 1, db = "db.t3.medium" }
    cold = { api = 0, db = "snapshot-only" }
  }
}
```

### DR Environment Scheduling

```bash
# Schedule DR environment to reduce costs
# Start DR environment daily at 8 AM for testing
0 8 * * * aws ec2 start-instances --instance-ids i-dr-instance

# Stop DR environment at 6 PM to save costs
0 18 * * * aws ec2 stop-instances --instance-ids i-dr-instance
```

## Compliance and Documentation

### Regulatory Requirements

- **SOC 2**: Maintain 99.9% availability SLA
- **ISO 27001**: Document all recovery procedures
- **GDPR**: Ensure data protection during recovery
- **HIPAA**: Maintain audit trails for healthcare data

### Audit Trail

```python
# audit_logger.py
def log_dr_activity(action, user, details):
    audit_entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'action': action,
        'user': user,
        'details': details,
        'compliance_flags': check_compliance(action)
    }
    
    # Log to secure audit system
    send_to_audit_log(audit_entry)
    
    # Alert on sensitive actions
    if action in ['data_restore', 'security_incident']:
        notify_compliance_team(audit_entry)
```