# SLSA (Supply Chain Levels for Software Artifacts) Compliance

This document outlines our SLSA compliance implementation for the AGI Evaluation Sandbox.

## SLSA Overview

SLSA (Supply Chain Levels for Software Artifacts) is a framework for securing software supply chains. We target **SLSA Level 2** compliance with pathways to Level 3.

### Current Compliance Status

| SLSA Level | Status | Target Date |
|------------|--------|-------------|
| Level 1 | ✅ Implemented | Completed |
| Level 2 | 🚧 In Progress | Q2 2024 |
| Level 3 | 📋 Planned | Q4 2024 |
| Level 4 | 🔮 Future | 2025 |

## SLSA Level 1 Requirements

### ✅ Version Control
- **Requirement**: Source code version controlled
- **Implementation**: Git repository with complete history
- **Evidence**: GitHub repository with commit history

### ✅ Build Service
- **Requirement**: Build service generates the package
- **Implementation**: GitHub Actions CI/CD pipeline
- **Evidence**: Workflow logs and build artifacts

```yaml
# .github/workflows/build.yml (Level 1)
name: Build and Package
on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0  # Full history for provenance
      
      - name: Build Application
        run: |
          npm run build
          docker build -t agi-eval-sandbox:${{ github.sha }} .
      
      - name: Generate Basic Provenance
        run: |
          echo "Built from commit: ${{ github.sha }}" > build-info.txt
          echo "Build time: $(date -u)" >> build-info.txt
```

## SLSA Level 2 Requirements

### 🚧 Build Service
- **Requirement**: Hosted build service (not self-hosted)
- **Implementation**: GitHub Actions (hosted runners only)
- **Configuration**:

```yaml
# .github/workflows/slsa-build.yml (Level 2)
name: SLSA Level 2 Build
on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

jobs:
  build:
    runs-on: ubuntu-latest  # Hosted runner required
    outputs:
      hashes: ${{ steps.hash.outputs.hashes }}
      image-digest: ${{ steps.build.outputs.digest }}
    
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Build and Hash
        id: build
        run: |
          # Build application
          npm run build
          docker build -t agi-eval-sandbox:${{ github.sha }} .
          
          # Generate hash of artifacts
          sha256sum dist/* > checksums.txt
          
          # Output for provenance
          echo "digest=$(docker inspect --format='{{index .RepoDigests 0}}' agi-eval-sandbox:${{ github.sha }})" >> $GITHUB_OUTPUT
      
      - name: Generate Hashes
        id: hash
        run: |
          echo "hashes=$(cat checksums.txt | base64 -w0)" >> $GITHUB_OUTPUT

  provenance:
    needs: [build]
    permissions:
      actions: read
      id-token: write
      contents: write
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
    with:
      base64-subjects: "${{ needs.build.outputs.hashes }}"
      upload-assets: true
```

### 🚧 Source Control Integration
- **Requirement**: Build config and build steps documented
- **Implementation**: Version-controlled build configuration

```dockerfile
# Multi-stage Dockerfile for reproducible builds
FROM node:18-alpine AS frontend-builder
WORKDIR /app/dashboard
COPY dashboard/package*.json ./
RUN npm ci --only=production
COPY dashboard/ ./
RUN npm run build

FROM python:3.11-slim AS backend-builder
WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .
COPY src/ ./src/
RUN python -m build

FROM python:3.11-slim AS runtime
# Copy built artifacts with verified checksums
COPY --from=backend-builder /app/dist/ /opt/agi-eval/
COPY --from=frontend-builder /app/dashboard/dist/ /opt/agi-eval/static/
```

### 🚧 Provenance Generation
- **Requirement**: Cryptographically signed provenance
- **Implementation**: SLSA provenance generator

```yaml
# Generate signed provenance
- name: Generate Provenance
  uses: slsa-framework/slsa-github-generator@v1.9.0
  with:
    base64-subjects: ${{ needs.build.outputs.hashes }}
    provenance-name: "agi-eval-sandbox.intoto.jsonl"
    upload-assets: true
```

## SLSA Level 3 Requirements (Planned)

### 📋 Isolated Build Environment
- **Requirement**: Build runs in isolated environment
- **Implementation**: Self-hosted runners with VM isolation

```yaml
# Future Level 3 implementation
jobs:
  build-level3:
    runs-on: [self-hosted, vm-isolated]
    container:
      image: ubuntu:22.04
      options: --isolation=hyperv  # Windows example
    
    steps:
      - name: Secure Build Environment
        run: |
          # Verify environment isolation
          systemctl status apparmor
          docker info | grep "Security Options"
```

### 📋 Dependency Tracking
- **Requirement**: All dependencies explicitly declared
- **Implementation**: Lock files and SBOM generation

```yaml
# Enhanced dependency tracking
- name: Generate SBOM
  run: |
    # Python dependencies
    pip-audit --format=cyclonedx --output=python-sbom.json
    
    # Node.js dependencies
    cyclonedx-npm --output-file=nodejs-sbom.json
    
    # Container dependencies
    syft docker:agi-eval-sandbox:${{ github.sha }} -o spdx-json > container-sbom.json

- name: Verify Dependencies
  run: |
    # Check for known vulnerabilities
    grype sbom:python-sbom.json
    grype sbom:nodejs-sbom.json
    grype sbom:container-sbom.json
```

## Build Provenance

### Provenance Schema

```json
{
  "_type": "https://in-toto.io/Statement/v0.1",
  "predicateType": "https://slsa.dev/provenance/v0.2",
  "subject": [
    {
      "name": "agi-eval-sandbox",
      "digest": {
        "sha256": "abc123..."
      }
    }
  ],
  "predicate": {
    "builder": {
      "id": "https://github.com/actions/runner"
    },
    "buildType": "https://github.com/actions/workflow",
    "invocation": {
      "configSource": {
        "uri": "git+https://github.com/your-org/agi-eval-sandbox",
        "digest": {
          "sha1": "def456..."
        }
      }
    },
    "buildConfig": {
      "steps": [
        {
          "command": ["npm", "run", "build"],
          "env": {
            "NODE_ENV": "production"
          }
        }
      ]
    },
    "materials": [
      {
        "uri": "git+https://github.com/your-org/agi-eval-sandbox",
        "digest": {
          "sha1": "def456..."
        }
      }
    ]
  }
}
```

### Provenance Verification

```bash
#!/bin/bash
# verify_provenance.sh

# Download provenance
gh release download v1.0.0 --pattern "*.intoto.jsonl"

# Verify signature
slsa-verifier verify-artifact \
  --provenance-path agi-eval-sandbox.intoto.jsonl \
  --source-uri github.com/your-org/agi-eval-sandbox \
  --source-tag v1.0.0 \
  agi-eval-sandbox-v1.0.0.tar.gz

# Check build parameters
jq '.predicate.buildConfig' agi-eval-sandbox.intoto.jsonl
```

## Supply Chain Security

### Dependency Management

```yaml
# .github/workflows/dependency-review.yml
name: Dependency Review
on:
  pull_request:
    paths:
      - 'package*.json'
      - 'pyproject.toml'
      - 'requirements*.txt'

jobs:
  dependency-review:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/dependency-review-action@v3
        with:
          fail-on-severity: high
          allow-licenses: MIT, Apache-2.0, BSD-3-Clause
```

### Container Image Signing

```yaml
# Sign container images with Cosign
- name: Install Cosign
  uses: sigstore/cosign-installer@v3

- name: Sign Container Image
  run: |
    cosign sign --yes docker.io/your-org/agi-eval-sandbox:${{ github.sha }}

- name: Generate SBOM and Attest
  run: |
    syft docker.io/your-org/agi-eval-sandbox:${{ github.sha }} -o spdx-json > sbom.json
    cosign attest --yes --predicate sbom.json docker.io/your-org/agi-eval-sandbox:${{ github.sha }}
```

## Verification Tools

### Custom Verification Script

```python
#!/usr/bin/env python3
# verify_slsa_compliance.py

import json
import subprocess
import sys
from pathlib import Path

def verify_provenance(artifact_path, provenance_path):
    """Verify SLSA provenance for an artifact."""
    try:
        # Verify using slsa-verifier
        result = subprocess.run([
            'slsa-verifier', 'verify-artifact',
            '--provenance-path', provenance_path,
            '--source-uri', 'github.com/your-org/agi-eval-sandbox',
            artifact_path
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Provenance verified for {artifact_path}")
            return True
        else:
            print(f"❌ Provenance verification failed: {result.stderr}")
            return False
    except FileNotFoundError:
        print("❌ slsa-verifier not found. Install from: https://github.com/slsa-framework/slsa-verifier")
        return False

def check_build_reproducibility(build_config):
    """Check if build is reproducible."""
    reproducible_factors = [
        'locked_dependencies',
        'pinned_base_images',
        'explicit_build_tools',
        'deterministic_build_order'
    ]
    
    score = 0
    for factor in reproducible_factors:
        if build_config.get(factor, False):
            score += 1
            print(f"✅ {factor}")
        else:
            print(f"❌ {factor}")
    
    print(f"Reproducibility score: {score}/{len(reproducible_factors)}")
    return score == len(reproducible_factors)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: verify_slsa_compliance.py <artifact> <provenance>")
        sys.exit(1)
    
    artifact_path = sys.argv[1]
    provenance_path = sys.argv[2]
    
    if verify_provenance(artifact_path, provenance_path):
        print("SLSA compliance verification passed!")
        sys.exit(0)
    else:
        print("SLSA compliance verification failed!")
        sys.exit(1)
```

## Monitoring and Metrics

### SLSA Compliance Dashboard

```python
# slsa_metrics.py
from prometheus_client import Counter, Histogram, Gauge

# Metrics for SLSA compliance
slsa_builds_total = Counter('slsa_builds_total', 'Total SLSA compliant builds', ['level'])
slsa_verification_time = Histogram('slsa_verification_seconds', 'Time to verify SLSA provenance')
slsa_compliance_score = Gauge('slsa_compliance_score', 'Current SLSA compliance score')

def record_slsa_build(level):
    """Record a SLSA compliant build."""
    slsa_builds_total.labels(level=level).inc()

def record_verification_time(duration):
    """Record provenance verification time."""
    slsa_verification_time.observe(duration)

def update_compliance_score(score):
    """Update overall compliance score."""
    slsa_compliance_score.set(score)
```

### Alerting Rules

```yaml
# prometheus-slsa-alerts.yml
groups:
  - name: slsa_compliance
    rules:
      - alert: SLSAComplianceDown
        expr: slsa_compliance_score < 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "SLSA compliance score below threshold"
          description: "Current SLSA compliance score is {{ $value }}"
      
      - alert: ProvenanceVerificationFailed
        expr: increase(slsa_verification_failures_total[1h]) > 0
        labels:
          severity: critical
        annotations:
          summary: "SLSA provenance verification failed"
          description: "{{ $value }} provenance verifications failed in the last hour"
```

## Integration with Existing Tools

### GitHub Security Features

```yaml
# .github/workflows/security-scanning.yml
name: Security Scanning
on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  codeql:
    runs-on: ubuntu-latest
    permissions:
      security-events: write
    steps:
      - uses: actions/checkout@v4
      - uses: github/codeql-action/init@v2
        with:
          languages: python, javascript
      - uses: github/codeql-action/analyze@v2

  dependency-scan:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Run safety check
        run: |
          pip install safety
          safety check --json --output safety-report.json
      - name: Upload to GitHub Security
        uses: github/codeql-action/upload-sarif@v2
        with:
          sarif_file: safety-report.json
```

### Integration with Policy as Code

```rego
# slsa_policy.rego (Open Policy Agent)
package slsa.build

# Deny builds that don't meet SLSA Level 2
deny[msg] {
    input.builder.id != "https://github.com/actions/runner"
    msg := "Build must use hosted GitHub Actions runner"
}

deny[msg] {
    count(input.materials) == 0
    msg := "Build materials must be declared"
}

deny[msg] {
    not input.predicate.buildType
    msg := "Build type must be specified"
}

# Allow builds that meet all requirements
allow {
    count(deny) == 0
}
```

## Future Roadmap

### SLSA Level 4 Planning

- **Hermetic Builds**: Complete isolation from external dependencies during build
- **Reproducible Builds**: Bit-for-bit reproducible artifacts
- **Two-Person Review**: All changes require two-person approval
- **Automated Policy Enforcement**: Policy as code for all security decisions

### Integration Opportunities

- **Sigstore Integration**: Keyless signing with Sigstore/Fulcio
- **Supply Chain Security Tools**: Integration with GUAC, Dependency-Track
- **Policy Frameworks**: Integration with Open Policy Agent, Falco
- **Attestation Standards**: Support for in-toto ITE-6 specifications