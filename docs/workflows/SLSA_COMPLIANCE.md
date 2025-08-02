# SLSA Compliance Guide

## Overview

Supply-chain Levels for Software Artifacts (SLSA) is a security framework for protecting software supply chains. This guide outlines how to implement SLSA compliance for the AGI Evaluation Sandbox.

## SLSA Requirements

### SLSA Level 1: Documentation of the build process
- ✅ Build scripts are version controlled
- ✅ Build process is documented
- ✅ Provenance is generated

### SLSA Level 2: Tamper resistance of the build service
- ✅ Hosted build service (GitHub Actions)
- ✅ Source integrity verification
- ✅ Authenticated provenance

### SLSA Level 3: Extra resistance to specific threats
- ✅ Security controls on host
- ✅ Non-falsifiable provenance
- ✅ Isolated builds

## Implementation

### 1. SLSA Generator Workflow

Create `.github/workflows/slsa.yml`:

```yaml
name: SLSA Provenance

on:
  push:
    tags: [ 'v*' ]
  workflow_dispatch:

permissions: read-all

jobs:
  # Build artifacts
  build:
    permissions:
      id-token: write # For signing
      contents: read  # For repo checkout
      actions: read   # For getting workflow run info
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_generic_slsa3.yml@v1.9.0
    with:
      base64-subjects: "${{ needs.provenance-subjects.outputs.hashes }}"
      upload-assets: true
      
  # Generate provenance subjects
  provenance-subjects:
    runs-on: ubuntu-latest
    outputs:
      hashes: ${{ steps.hash.outputs.hashes }}
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Build artifacts
        run: |
          make build
          
      - name: Generate hashes
        shell: bash
        id: hash
        run: |
          set -euo pipefail
          
          # Generate checksums for all build artifacts
          (cd dist && find . -type f -name "*.whl" -o -name "*.tar.gz" | xargs sha256sum | base64 -w0)
```

### 2. Container Image SLSA Provenance

Create `.github/workflows/slsa-container.yml`:

```yaml
name: SLSA Container Provenance

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

permissions: read-all

jobs:
  # Build container with SLSA provenance
  build:
    permissions:
      id-token: write   # For signing
      packages: write   # For pushing to GHCR
      contents: read    # For repo checkout
      actions: read     # For getting workflow run info
    uses: slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@v1.9.0
    with:
      image: ghcr.io/${{ github.repository }}
      digest: ${{ needs.image.outputs.digest }}
      registry-username: ${{ github.actor }}
    secrets:
      registry-password: ${{ secrets.GITHUB_TOKEN }}
```

### 3. Verification Tools

#### Verify SLSA Provenance

```bash
#!/bin/bash
# verify-slsa.sh

ARTIFACT="$1"
PROVENANCE="$2"

# Install slsa-verifier
go install github.com/slsa-framework/slsa-verifier/v2/cli/slsa-verifier@latest

# Verify the provenance
slsa-verifier verify-artifact \
  --provenance-path "$PROVENANCE" \
  --source-uri github.com/your-org/agi-eval-sandbox \
  "$ARTIFACT"

echo "SLSA verification completed"
```

#### Verify Container Images

```bash
#!/bin/bash
# verify-container.sh

IMAGE="$1"

# Install cosign
go install github.com/sigstore/cosign/v2/cmd/cosign@latest

# Verify container signature and SLSA provenance
cosign verify \
  --certificate-identity-regexp "^https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@refs/tags/v[0-9]+.[0-9]+.[0-9]+$" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  "$IMAGE"

# Verify SLSA provenance
cosign verify-attestation \
  --type slsaprovenance \
  --certificate-identity-regexp "^https://github.com/slsa-framework/slsa-github-generator/.github/workflows/generator_container_slsa3.yml@refs/tags/v[0-9]+.[0-9]+.[0-9]+$" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  "$IMAGE"

echo "Container verification completed"
```

## SBOM Generation

### 1. Software Bill of Materials Workflow

Create `.github/workflows/sbom.yml`:

```yaml
name: SBOM Generation

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]
  pull_request:
    branches: [ main ]

jobs:
  generate-sbom:
    name: Generate SBOM
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
      id-token: write
      
    steps:
      - name: Checkout code
        uses: actions/checkout@v4
        
      - name: Build Docker image
        run: |
          docker build -t agi-eval-sandbox:sbom .
          
      - name: Generate SBOM with Syft
        uses: anchore/sbom-action@v0
        with:
          image: agi-eval-sandbox:sbom
          format: spdx-json
          output-file: sbom.spdx.json
          
      - name: Generate SBOM with CycloneDX
        uses: CycloneDX/gh-python-generate-sbom@v1
        with:
          input: ./requirements.txt
          output: sbom.cyclonedx.json
          format: json
          
      - name: Sign SBOM with cosign
        uses: sigstore/cosign-installer@v3
        
      - name: Sign SBOMs
        run: |
          # Sign SPDX SBOM
          cosign sign-blob --yes sbom.spdx.json --output-signature sbom.spdx.json.sig
          
          # Sign CycloneDX SBOM
          cosign sign-blob --yes sbom.cyclonedx.json --output-signature sbom.cyclonedx.json.sig
          
      - name: Upload SBOMs
        uses: actions/upload-artifact@v3
        with:
          name: sboms
          path: |
            sbom.spdx.json
            sbom.spdx.json.sig
            sbom.cyclonedx.json
            sbom.cyclonedx.json.sig
            
      - name: Upload SBOM to dependency graph
        uses: advanced-security/spdx-dependency-submission-action@v0.0.1
        with:
          filePath: sbom.spdx.json
```

### 2. SBOM Verification

```bash
#!/bin/bash
# verify-sbom.sh

SBOM_FILE="$1"
SIGNATURE_FILE="$2"

# Verify SBOM signature
cosign verify-blob \
  --certificate-identity-regexp "^https://github.com/your-org/agi-eval-sandbox/.github/workflows/sbom.yml@refs/heads/main$" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  --signature "$SIGNATURE_FILE" \
  "$SBOM_FILE"

echo "SBOM verification completed"
```

## Security Scanning Integration

### 1. Vulnerability Scanning

```yaml
# .github/workflows/vulnerability-scan.yml
name: Vulnerability Scan

on:
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM
  workflow_dispatch:

jobs:
  scan-sbom:
    name: Scan SBOM for Vulnerabilities
    runs-on: ubuntu-latest
    
    steps:
      - name: Download latest SBOM
        run: |
          # Download SBOM from latest release
          gh release download --pattern "sbom.*" --repo ${{ github.repository }}
          
      - name: Scan SBOM with Grype
        uses: anchore/scan-action@v3
        with:
          sbom: sbom.spdx.json
          fail-build: false
          severity-cutoff: high
          
      - name: Upload vulnerability report
        uses: actions/upload-artifact@v3
        with:
          name: vulnerability-report
          path: results.sarif
```

## Compliance Reporting

### 1. SLSA Compliance Dashboard

Create a compliance dashboard that shows:

- SLSA level achieved
- Provenance verification status
- SBOM generation status
- Vulnerability scan results
- Supply chain security posture

### 2. Compliance Metrics

Track the following metrics:

```yaml
# Example metrics
slsa_compliance:
  level: 3
  provenance_generated: true
  sbom_available: true
  signatures_verified: true
  last_verification: "2024-01-15T10:30:00Z"
  
supply_chain_security:
  vulnerabilities:
    critical: 0
    high: 1
    medium: 3
    low: 12
  dependencies_scanned: true
  licenses_compliant: true
  secrets_detected: false
```

## Best Practices

### 1. Build Environment Security

- Use hosted runners (GitHub-hosted)
- Minimize build dependencies
- Use pinned versions for all tools
- Implement least privilege access

### 2. Artifact Integrity

- Sign all artifacts with cosign
- Generate checksums for all releases
- Use content-addressable storage
- Implement artifact verification

### 3. Provenance Management

- Generate provenance for all builds
- Store provenance with artifacts
- Implement provenance verification
- Use tamper-evident storage

### 4. Dependency Management

- Pin all dependencies with checksums
- Regular dependency updates
- Vulnerability scanning
- License compliance checking

## Verification Commands

### For Users/Consumers

```bash
# Verify a release artifact
slsa-verifier verify-artifact \
  --provenance-path agi-eval-sandbox.intoto.jsonl \
  --source-uri github.com/your-org/agi-eval-sandbox \
  agi-eval-sandbox-1.0.0.tar.gz

# Verify container image
cosign verify \
  --certificate-identity-regexp "^https://github.com/slsa-framework/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  ghcr.io/your-org/agi-eval-sandbox:v1.0.0

# Verify SBOM
cosign verify-blob \
  --certificate-identity-regexp "^https://github.com/your-org/.*" \
  --certificate-oidc-issuer "https://token.actions.githubusercontent.com" \
  --signature sbom.spdx.json.sig \
  sbom.spdx.json
```

This SLSA compliance implementation provides strong supply chain security guarantees and enables users to verify the integrity and provenance of all artifacts.