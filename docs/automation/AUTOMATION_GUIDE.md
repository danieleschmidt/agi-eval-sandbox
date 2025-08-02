# Automation Guide

## Overview

This guide covers the comprehensive automation setup for the AGI Evaluation Sandbox, including metrics collection, repository maintenance, and integration automation.

## Automation Components

### 1. Metrics Collection System

The metrics collection system automatically gathers project health indicators.

#### Automated Metrics Collection

```bash
# Run metrics collection
python scripts/metrics-collection.py

# Save to specific file
python scripts/metrics-collection.py --output metrics/daily-report.json

# Quiet mode for automation
python scripts/metrics-collection.py --quiet --output metrics/$(date +%Y%m%d).json
```

#### Collected Metrics Categories

**Development Metrics:**
- Git repository statistics (commits, contributors, activity)
- Code quality metrics (lines of code, test coverage, complexity)
- Dependency information (count, security status)
- Documentation coverage

**Security Metrics:**
- Security files presence
- Potential secrets detection
- Vulnerability scan results
- License compliance

**Automation Metrics:**
- GitHub Actions workflows
- Pre-commit hooks configuration
- Build automation (Makefile, Docker)
- CI/CD pipeline health

#### Metrics Configuration

The metrics are defined in `.github/project-metrics.json`:

```json
{
  "metrics": {
    "development": {
      "code_quality": {
        "test_coverage": {
          "target": 80,
          "current": 0,
          "unit": "percentage",
          "trend": "stable"
        }
      }
    }
  }
}
```

### 2. Repository Maintenance Automation

Automated maintenance tasks to keep the repository healthy.

#### Repository Maintenance Script

```bash
# Dry run (default) - shows what would be done
python scripts/repository-maintenance.py

# Execute maintenance tasks
python scripts/repository-maintenance.py --execute

# Save detailed report
python scripts/repository-maintenance.py --output maintenance-report.json
```

#### Maintenance Tasks

**Security Maintenance:**
- Vulnerability scanning (Python and Node.js)
- License compliance checking
- Configuration file validation
- Secrets detection

**Dependency Management:**
- Dependency update checking
- Security vulnerability assessment
- License compatibility verification
- Dependency graph analysis

**Repository Cleanup:**
- Cache file cleanup
- Old branch removal
- Temporary file cleanup
- Build artifact cleanup

**Configuration Validation:**
- JSON/YAML syntax validation
- Docker configuration verification
- CI/CD workflow validation

### 3. Dependency Update Automation

Automated dependency updates with testing and PR creation.

#### Dependency Update Script

```bash
# Check what needs updating (dry run)
./scripts/dependency-update-automation.sh --dry-run

# Perform automated updates
./scripts/dependency-update-automation.sh

# Environment variable control
DRY_RUN=true ./scripts/dependency-update-automation.sh
```

#### Update Process

1. **Prerequisite Check:** Verifies required tools (git, gh, npm, pip)
2. **Dependency Analysis:** Identifies outdated packages
3. **Automated Updates:** Updates packages using appropriate tools
4. **Testing:** Runs full test suite to verify compatibility
5. **PR Creation:** Creates pull request with detailed information
6. **Cleanup:** Reverts changes if tests fail

#### Update Features

- **Smart Batching:** Groups related updates
- **Test Validation:** Ensures updates don't break functionality
- **Rollback Capability:** Reverts on test failures
- **Detailed Reporting:** Comprehensive update documentation

### 4. CI/CD Integration

Automation integrated into the CI/CD pipeline.

#### GitHub Actions Integration

**Metrics Collection Workflow:**

```yaml
name: Collect Metrics
on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM
  workflow_dispatch:

jobs:
  collect-metrics:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Collect metrics
        run: python scripts/metrics-collection.py --output metrics/$(date +%Y%m%d).json
      - name: Upload metrics
        uses: actions/upload-artifact@v3
        with:
          name: metrics
          path: metrics/
```

**Maintenance Workflow:**

```yaml
name: Repository Maintenance
on:
  schedule:
    - cron: '0 2 * * 1'  # Weekly on Monday at 2 AM
  workflow_dispatch:

jobs:
  maintenance:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -r requirements-dev.txt
      - name: Run maintenance
        run: python scripts/repository-maintenance.py --execute
      - name: Create issue if problems found
        if: failure()
        uses: actions/github-script@v6
        with:
          script: |
            github.rest.issues.create({
              owner: context.repo.owner,
              repo: context.repo.repo,
              title: 'Repository Maintenance Issues Detected',
              body: 'Automated maintenance found issues that need attention.',
              labels: ['maintenance', 'automated']
            });
```

**Dependency Update Workflow:**

```yaml
name: Dependency Updates
on:
  schedule:
    - cron: '0 4 * * 1'  # Weekly on Monday at 4 AM
  workflow_dispatch:

jobs:
  update-dependencies:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          token: ${{ secrets.GITHUB_TOKEN }}
      - name: Setup environment
        run: |
          # Setup Python and Node.js
          # Install dependencies
      - name: Run dependency updates
        run: ./scripts/dependency-update-automation.sh
```

### 5. Monitoring and Alerting Automation

Automated monitoring with intelligent alerting.

#### Health Check Automation

```bash
# Basic health check
curl -f https://api.agi-eval.com/health

# Comprehensive health check
python scripts/health-check.py --comprehensive --alert-on-failure
```

#### Alerting Configuration

**Slack Integration:**

```python
import requests

def send_slack_alert(message, channel="#alerts"):
    webhook_url = os.getenv("SLACK_WEBHOOK_URL")
    payload = {
        "channel": channel,
        "text": message,
        "username": "AGI-Eval-Bot"
    }
    requests.post(webhook_url, json=payload)
```

**Email Alerts:**

```python
import smtplib
from email.mime.text import MIMEText

def send_email_alert(subject, body, recipients):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = 'alerts@agi-eval.com'
    msg['To'] = ', '.join(recipients)
    
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(username, password)
    server.send_message(msg)
    server.quit()
```

### 6. Performance Automation

Automated performance monitoring and optimization.

#### Performance Testing

```bash
# Run automated performance tests
npm run test:performance

# Load testing with k6
k6 run performance/k6/load-test.js

# Lighthouse CI for web performance
npx lhci autorun
```

#### Performance Optimization

```python
# Automated database optimization
def optimize_database():
    # Analyze slow queries
    # Update indexes
    # Vacuum and analyze
    pass

# Cache warming
def warm_caches():
    # Pre-populate frequently accessed data
    # Warm Redis caches
    # Pre-generate static content
    pass
```

### 7. Backup and Recovery Automation

Automated backup and disaster recovery procedures.

#### Database Backup

```bash
#!/bin/bash
# Automated database backup
pg_dump $DATABASE_URL | gzip > backups/db-$(date +%Y%m%d-%H%M%S).sql.gz

# Upload to cloud storage
aws s3 cp backups/ s3://agi-eval-backups/ --recursive
```

#### Configuration Backup

```python
import shutil
import datetime

def backup_configurations():
    """Backup critical configuration files."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_dir = f"backups/config-{timestamp}"
    
    # Backup critical files
    files_to_backup = [
        "docker-compose.yml",
        "nginx/nginx.conf",
        "monitoring/prometheus.yml",
        ".github/workflows/"
    ]
    
    for file_path in files_to_backup:
        shutil.copytree(file_path, f"{backup_dir}/{file_path}")
```

### 8. Deployment Automation

Zero-downtime deployment automation.

#### Blue-Green Deployment

```bash
#!/bin/bash
# Blue-green deployment script

CURRENT_ENV=$(kubectl get service app-service -o jsonpath='{.spec.selector.version}')
NEW_ENV=$([[ "$CURRENT_ENV" == "blue" ]] && echo "green" || echo "blue")

# Deploy new version
kubectl set image deployment/app-$NEW_ENV app=myapp:$NEW_VERSION

# Wait for deployment
kubectl rollout status deployment/app-$NEW_ENV

# Run health checks
if curl -f http://app-$NEW_ENV/health; then
    # Switch traffic
    kubectl patch service app-service -p '{"spec":{"selector":{"version":"'$NEW_ENV'"}}}'
    
    # Cleanup old deployment
    kubectl delete deployment app-$CURRENT_ENV
else
    echo "Health check failed, rolling back"
    kubectl delete deployment app-$NEW_ENV
    exit 1
fi
```

#### Canary Deployment

```yaml
# Canary deployment with Istio
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: app-canary
spec:
  http:
  - match:
    - headers:
        canary:
          exact: "true"
    route:
    - destination:
        host: app-service
        subset: canary
  - route:
    - destination:
        host: app-service
        subset: stable
      weight: 90
    - destination:
        host: app-service
        subset: canary
      weight: 10
```

## Automation Best Practices

### 1. Error Handling

```python
import logging
from functools import wraps

def retry_on_failure(max_retries=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise
                    logging.warning(f"Attempt {attempt + 1} failed: {e}")
                    time.sleep(delay * (2 ** attempt))  # Exponential backoff
            return wrapper
        return decorator

@retry_on_failure(max_retries=3)
def unreliable_operation():
    # Operation that might fail
    pass
```

### 2. Logging and Monitoring

```python
import structlog

logger = structlog.get_logger()

def automated_task():
    logger.info("Starting automated task", task="metrics_collection")
    
    try:
        # Perform task
        result = collect_metrics()
        logger.info("Task completed successfully", 
                   task="metrics_collection", 
                   metrics_count=len(result))
    except Exception as e:
        logger.error("Task failed", 
                    task="metrics_collection", 
                    error=str(e),
                    exc_info=True)
        raise
```

### 3. Configuration Management

```python
from pydantic import BaseSettings

class AutomationConfig(BaseSettings):
    slack_webhook_url: str
    github_token: str
    metrics_retention_days: int = 30
    max_retries: int = 3
    
    class Config:
        env_file = ".env"

config = AutomationConfig()
```

### 4. Testing Automation

```python
import pytest
from unittest.mock import patch

def test_metrics_collection():
    """Test automated metrics collection."""
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = "test output"
        
        collector = MetricsCollector()
        metrics = collector.collect_git_metrics()
        
        assert metrics is not None
        assert "commit_count" in metrics
```

## Maintenance Schedule

### Daily Automation (6:00 AM UTC)
- Metrics collection
- Health checks
- Security scanning
- Performance monitoring

### Weekly Automation (Monday 2:00 AM UTC)
- Dependency updates
- Repository maintenance
- License compliance check
- Backup verification

### Monthly Automation (1st of month)
- Comprehensive security audit
- Performance optimization
- Configuration backup
- Documentation updates

### Quarterly Automation
- Disaster recovery testing
- Capacity planning analysis
- Technology stack review
- Automation improvement review

This comprehensive automation system ensures the AGI Evaluation Sandbox maintains high quality, security, and performance with minimal manual intervention.