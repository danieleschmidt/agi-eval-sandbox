# Deployment Security Guidelines

## Overview

This document outlines security requirements and best practices for deploying the AGI Evaluation Sandbox, focusing on container security, secrets management, and secure CI/CD practices.

## Container Security

### Image Security Standards

#### Base Image Requirements
```dockerfile
# Use minimal, security-hardened base images
FROM python:3.11-slim-bullseye

# Run as non-root user
RUN groupadd -r appuser && useradd -r -g appuser appuser
USER appuser

# Set secure file permissions
COPY --chown=appuser:appuser . /app
WORKDIR /app
```

#### Security Scanning Pipeline
```yaml
# security-scanning.yml
image_scanning:
  tools:
    - trivy      # Vulnerability scanning
    - hadolint   # Dockerfile linting  
    - dockle     # Container security
    - grype      # Vulnerability scanner
    
  policies:
    block_high_severity: true
    block_critical_severity: true
    allow_medium_with_fix: true
    
  compliance:
    - cis_docker_benchmark
    - nist_800_190
```

### Container Hardening

#### Security Context
```yaml
# k8s-security-context.yml
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  
  capabilities:
    drop:
      - ALL
    add:
      - NET_BIND_SERVICE  # Only if needed
      
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  
  seccompProfile:
    type: RuntimeDefault
    
  seLinuxOptions:
    level: "s0:c123,c456"
```

#### Resource Limits
```yaml
# resource-security.yml
resources:
  requests:
    cpu: "100m"
    memory: "256Mi"
  limits:
    cpu: "1000m"
    memory: "1Gi"
    ephemeral-storage: "1Gi"
    
# Prevent resource exhaustion
podDisruptionBudget:
  minAvailable: 1
  
# Limit concurrent pods
maxUnavailable: 25%
maxSurge: 25%
```

## Secrets Management

### Secret Classification

| Type | Sensitivity | Storage | Rotation | Access |
|------|-------------|---------|----------|---------|
| **API Keys** | High | External vault | 30 days | Service accounts only |
| **Database Credentials** | Critical | Kubernetes secrets | 90 days | Database admin + apps |
| **TLS Certificates** | High | cert-manager | 90 days | Ingress controllers |
| **Signing Keys** | Critical | HSM/KMS | 1 year | CI/CD system only |

### Secret Store Configuration

#### Kubernetes Secrets
```yaml
# secrets-config.yml
apiVersion: v1
kind: Secret
metadata:
  name: agi-eval-secrets
  namespace: agi-eval
type: Opaque
stringData:
  database-url: "postgresql://user:pass@host:5432/db"
  openai-api-key: "sk-..."
  anthropic-api-key: "claude-..."
  
# Enable encryption at rest
---
apiVersion: v1
kind: EncryptionConfiguration
resources:
  - resources: ["secrets"]
    providers:
      - aescbc:
          keys:
            - name: key1
              secret: <base64-encoded-key>
```

#### External Secret Management
```yaml
# external-secrets.yml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.your-org.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "agi-eval-role"
```

### Secret Injection Patterns

#### Environment Variables (Least Secure)
```yaml
# Only for non-sensitive configuration
env:
  - name: LOG_LEVEL
    value: "INFO"
  - name: API_TIMEOUT
    value: "30"
```

#### Volume Mounts (Recommended)
```yaml
# Mount secrets as files
volumeMounts:
  - name: secrets-volume
    mountPath: "/etc/secrets"
    readOnly: true
    
volumes:
  - name: secrets-volume
    secret:
      secretName: agi-eval-secrets
      defaultMode: 0400  # Read-only for owner
```

#### Init Containers (Most Secure)
```yaml
# Fetch secrets in init container
initContainers:
  - name: secret-fetcher
    image: vault:latest
    command:
      - sh
      - -c
      - |
        vault auth -method=kubernetes
        vault kv get -field=api-key secret/agi-eval > /shared/api-key
    volumeMounts:
      - name: shared-secrets
        mountPath: /shared
```

## Network Security

### Network Policies

#### Ingress Controls
```yaml
# network-policy-ingress.yml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: agi-eval-ingress
spec:
  podSelector:
    matchLabels:
      app: agi-eval-sandbox
  policyTypes:
    - Ingress
    - Egress
  ingress:
    # Allow traffic from load balancer
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8000
    
    # Allow monitoring traffic
    - from:
        - namespaceSelector:
            matchLabels:
              name: monitoring
      ports:
        - protocol: TCP
          port: 9090
```

#### Egress Controls
```yaml
# network-policy-egress.yml  
egress:
  # Allow DNS resolution
  - to: []
    ports:
      - protocol: UDP
        port: 53
        
  # Allow database access
  - to:
      - namespaceSelector:
          matchLabels:
            name: database
    ports:
      - protocol: TCP
        port: 5432
        
  # Allow external API calls (restricted)
  - to: []
    ports:
      - protocol: TCP
        port: 443
    # Add specific IP/domain restrictions as needed
```

### TLS Configuration

#### Certificate Management
```yaml
# cert-manager.yml
apiVersion: cert-manager.io/v1
kind: Certificate
metadata:
  name: agi-eval-tls
spec:
  secretName: agi-eval-tls-secret
  issuer:
    name: letsencrypt-prod
    kind: ClusterIssuer
  dnsNames:
    - agi-eval.your-org.com
    - api.agi-eval.your-org.com
```

#### TLS Policy
```yaml
# tls-policy.yml
tls:
  minimum_version: "1.2"
  cipher_suites:
    - "TLS_ECDHE_RSA_WITH_AES_256_GCM_SHA384"
    - "TLS_ECDHE_RSA_WITH_AES_128_GCM_SHA256"
    - "TLS_ECDHE_RSA_WITH_CHACHA20_POLY1305"
  
  hsts:
    enabled: true
    max_age: 31536000
    include_subdomains: true
    preload: true
```

## Identity and Access Management

### Service Account Security

#### RBAC Configuration
```yaml
# rbac.yml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: agi-eval-sa
  namespace: agi-eval
  
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: agi-eval-role
rules:
  - apiGroups: [""]
    resources: ["secrets", "configmaps"]
    verbs: ["get", "list"]
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
    
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: agi-eval-binding
subjects:
  - kind: ServiceAccount
    name: agi-eval-sa
    namespace: agi-eval
roleRef:
  kind: Role
  name: agi-eval-role
  apiGroup: rbac.authorization.k8s.io
```

#### Pod Security Standards
```yaml
# pod-security.yml
apiVersion: v1
kind: Namespace
metadata:
  name: agi-eval
  labels:
    pod-security.kubernetes.io/enforce: restricted
    pod-security.kubernetes.io/audit: restricted
    pod-security.kubernetes.io/warn: restricted
```

### Authentication & Authorization

#### JWT Token Validation
```python
# Application-level security
from fastapi import HTTPException, Depends
from fastapi.security import HTTPBearer
import jwt

security = HTTPBearer()

async def verify_token(token: str = Depends(security)):
    try:
        payload = jwt.decode(
            token.credentials,
            key=get_public_key(),
            algorithms=["RS256"],
            audience="agi-eval-api"
        )
        return payload
    except jwt.InvalidTokenError:
        raise HTTPException(401, "Invalid token")

# Secure endpoint
@app.get("/api/secure")
async def secure_endpoint(user: dict = Depends(verify_token)):
    return {"user": user["sub"]}
```

#### API Rate Limiting
```yaml
# rate-limiting.yml
apiVersion: networking.istio.io/v1alpha3
kind: EnvoyFilter
metadata:
  name: rate-limit
spec:
  configPatches:
    - applyTo: HTTP_FILTER
      match:
        context: SIDECAR_INBOUND
      patch:
        operation: INSERT_BEFORE
        value:
          name: envoy.filters.http.local_ratelimit
          typed_config:
            "@type": type.googleapis.com/udpa.type.v1.TypedStruct
            type_url: type.googleapis.com/envoy.extensions.filters.http.local_ratelimit.v3.LocalRateLimit
            value:
              stat_prefix: rate_limiter
              token_bucket:
                max_tokens: 100
                tokens_per_fill: 10
                fill_interval: 60s
```

## CI/CD Security

### Pipeline Security

#### Secure Build Process
```yaml
# secure-build.yml
security_gates:
  pre_build:
    - dependency_check
    - license_scan
    - secret_detection
    
  build:
    - signed_commits_only
    - reproducible_builds
    - minimal_privileges
    
  post_build:
    - image_scanning
    - sbom_generation
    - signature_verification
    
  deployment:
    - admission_controllers
    - policy_enforcement
    - runtime_monitoring
```

#### Supply Chain Security
```yaml
# supply-chain.yml
supply_chain:
  source_verification:
    - signed_commits: required
    - branch_protection: true
    - required_reviews: 2
    
  dependency_management:
    - dependency_pinning: true
    - vulnerability_scanning: true
    - license_compliance: true
    
  artifact_security:
    - image_signing: true
    - sbom_generation: true
    - provenance_tracking: true
```

### Deployment Gates

#### Security Policies
```yaml
# opa-policies.yml
package kubernetes.admission

deny[msg] {
    input.request.kind.kind == "Pod"
    input.request.object.spec.securityContext.runAsRoot == true
    msg := "Pods must not run as root"
}

deny[msg] {
    input.request.kind.kind == "Pod"
    container := input.request.object.spec.containers[_]
    not container.securityContext.readOnlyRootFilesystem
    msg := "Containers must have read-only root filesystem"
}
```

#### Admission Controllers
```yaml
# admission-controllers.yml
admission_controllers:
  - PodSecurityPolicy
  - ImagePolicyWebhook
  - NamespaceLifecycle
  - ResourceQuota
  - LimitRanger
  - ServiceAccount
  - SecurityContextDeny
  - AlwaysPullImages
```

## Runtime Security

### Security Monitoring

#### Falco Rules
```yaml
# falco-rules.yml
- rule: Unauthorized Process in Container
  desc: Detect processes not in allowed list
  condition: >
    spawned_process and
    container and
    not proc.name in (python, gunicorn, nginx)
  output: >
    Unauthorized process in container
    (user=%user.name command=%proc.cmdline container_id=%container.id)
  priority: WARNING
  
- rule: Write to /etc Directory
  desc: Detect writes to /etc directory
  condition: >
    open_write and
    fd.name startswith /etc and
    container
  output: >
    Write to /etc directory
    (file=%fd.name user=%user.name container_id=%container.id)
  priority: ERROR
```

#### Security Alerts
```yaml
# security-alerting.yml
alerts:
  critical:
    - unauthorized_privilege_escalation
    - suspicious_network_activity
    - malware_detection
    - data_exfiltration_attempt
    
  warning:
    - failed_authentication_attempts
    - unusual_process_execution
    - configuration_changes
    - resource_consumption_anomalies
    
  notification_channels:
    - slack: "#security-alerts"
    - email: ["security-team@your-org.com"]
    - pagerduty: "security-incidents"
```

### Vulnerability Management

#### Scanning Schedule
```yaml
# vulnerability-management.yml
scanning:
  frequency:
    images: "daily"
    dependencies: "weekly"
    infrastructure: "monthly"
    
  severity_response:
    critical: "4 hours"
    high: "24 hours"
    medium: "7 days"
    low: "30 days"
    
  exceptions:
    process: "security_team_approval"
    max_duration: "90 days"
    documentation: "required"
```

## Compliance Framework

### Security Standards

#### Compliance Mapping
| Standard | Requirements | Implementation |
|----------|-------------|----------------|
| **SOC 2 Type II** | Access controls, encryption | RBAC, TLS, secrets management |
| **ISO 27001** | ISMS, risk management | Security policies, monitoring |
| **NIST CSF** | Identify, protect, detect | Security controls, incident response |
| **CIS Controls** | Basic security hygiene | Hardening, patching, monitoring |

#### Audit Requirements
```yaml
# audit-config.yml
audit:
  logging:
    api_requests: true
    authentication_events: true
    authorization_decisions: true
    configuration_changes: true
    
  retention:
    security_logs: "7 years"
    access_logs: "2 years"
    audit_trails: "10 years"
    
  compliance_reporting:
    frequency: "quarterly"
    automated: true
    external_validation: true
```

## Incident Response

### Security Incident Workflow

```mermaid
graph TD
    A[Security Alert] --> B[Triage]
    B --> C{Severity?}
    C -->|Low| D[Log & Monitor]
    C -->|Medium| E[Investigate]
    C -->|High| F[Immediate Response]
    C -->|Critical| G[Emergency Response]
    F --> H[Contain Threat]
    G --> H
    H --> I[Eradicate Threat]
    I --> J[Recover Systems]
    J --> K[Lessons Learned]
```

### Response Procedures

#### Immediate Actions
```bash
# Emergency response scripts
#!/bin/bash
# incident-response.sh

case $SEVERITY in
  "critical")
    # Isolate affected systems
    kubectl patch networkpolicy default-deny --type='merge' -p='{"spec":{"ingress":[]}}'
    
    # Scale down vulnerable deployments
    kubectl scale deployment agi-eval-sandbox --replicas=0
    
    # Alert security team
    curl -X POST "$SLACK_WEBHOOK" -d '{"text":"CRITICAL: Security incident detected"}'
    ;;
    
  "high")
    # Enable enhanced monitoring
    kubectl patch deployment agi-eval-sandbox -p='{"spec":{"template":{"spec":{"containers":[{"name":"api","env":[{"name":"LOG_LEVEL","value":"DEBUG"}]}]}}}}'
    
    # Rotate secrets
    kubectl delete secret agi-eval-secrets
    kubectl create secret generic agi-eval-secrets --from-env-file=new-secrets.env
    ;;
esac
```

## Security Checklist

### Pre-Deployment Security Review

#### Infrastructure Security
- [ ] Container images scanned and approved
- [ ] Network policies configured and tested
- [ ] Secrets properly managed and encrypted
- [ ] RBAC roles follow least privilege principle
- [ ] TLS certificates valid and configured
- [ ] Security contexts properly configured

#### Application Security  
- [ ] Authentication mechanisms implemented
- [ ] Authorization controls tested
- [ ] Input validation in place
- [ ] Output encoding configured
- [ ] Error handling secure (no info disclosure)
- [ ] Logging configured for security events

#### Operational Security
- [ ] Monitoring and alerting configured
- [ ] Incident response procedures tested
- [ ] Backup and recovery verified
- [ ] Security policies documented
- [ ] Compliance requirements met
- [ ] Vulnerability management process active

### Post-Deployment Validation

#### Security Testing
```bash
# Security validation scripts
#!/bin/bash

# Test authentication
curl -f https://api.agi-eval.your-org.com/health
curl -X POST https://api.agi-eval.your-org.com/api/secure

# Test TLS configuration
sslyze --regular api.agi-eval.your-org.com

# Test for common vulnerabilities
nmap -sV --script vuln api.agi-eval.your-org.com

# Validate network policies
kubectl auth can-i create pods --as=system:serviceaccount:default:default
```

## Related Documentation

- [Deployment Automation](DEPLOYMENT_AUTOMATION.md)
- [Infrastructure Security](../operational/INFRASTRUCTURE_SECURITY.md)
- [Incident Response Plan](../operational/INCIDENT_RESPONSE.md)
- [Compliance Framework](COMPLIANCE.md)

---

**Note**: This document provides security guidelines for deployment. Actual implementation must be adapted to your specific infrastructure, threat model, and compliance requirements. Regular security reviews and updates are essential.