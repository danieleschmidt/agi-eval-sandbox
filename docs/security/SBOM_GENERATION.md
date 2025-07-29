# Software Bill of Materials (SBOM) Generation Guide

This document outlines how to generate and maintain SBOMs for the AGI Evaluation Sandbox.

## Overview

Software Bill of Materials (SBOM) documents provide transparency into the components and dependencies used in our software, supporting security, compliance, and supply chain management.

## SBOM Generation Tools

### Python Dependencies (SPDX Format)

```bash
# Install SBOM generation tools
pip install pip-licenses cyclonedx-bom

# Generate SPDX SBOM for Python dependencies
pip-licenses --format=json --output-file=sbom-python-deps.json

# Generate CycloneDX SBOM
cyclonedx-py -o sbom-python-cyclonedx.json
```

### Node.js Dependencies (SPDX Format)

```bash
# Install SBOM generation tools
npm install -g @cyclonedx/cyclonedx-npm

# Generate CycloneDX SBOM for Node.js dependencies
cyclonedx-npm --output-file sbom-nodejs-cyclonedx.json

# Generate SPDX format
npx @spdx/spdx-sbom-generator -p package.json -o sbom-nodejs-spdx.json
```

## Automated SBOM Generation

### GitHub Actions Integration

Add to your CI/CD pipeline:

```yaml
- name: Generate Python SBOM
  run: |
    pip install cyclonedx-bom
    cyclonedx-py -o sbom-python.json
    
- name: Generate Node.js SBOM
  run: |
    npm install -g @cyclonedx/cyclonedx-npm
    cyclonedx-npm --output-file sbom-nodejs.json

- name: Upload SBOMs as artifacts
  uses: actions/upload-artifact@v3
  with:
    name: sbom-files
    path: |
      sbom-python.json
      sbom-nodejs.json
```

### Container SBOM Generation

```bash
# Generate SBOM for Docker image
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  anchore/syft:latest agi-eval-sandbox:latest -o spdx-json > sbom-container.json

# Generate SBOM with Trivy
trivy image --format spdx-json --output sbom-trivy.json agi-eval-sandbox:latest
```

## SBOM Content Standards

### Required Components

1. **Package Information**
   - Name, version, supplier
   - Download location
   - License information
   - Copyright notices

2. **Dependency Relationships**
   - Direct dependencies
   - Transitive dependencies
   - Dependency graph structure

3. **Security Metadata**
   - Known vulnerabilities
   - Security assessments
   - Patch status

### SBOM Formats Supported

- **SPDX** (Software Package Data Exchange)
- **CycloneDX** (OWASP standard)
- **SWID** (Software Identification tags)

## Vulnerability Integration

### SBOM + CVE Mapping

```bash
# Scan SBOM for vulnerabilities
grype sbom:sbom-python.json -o json > vulnerability-report.json

# OSV scanner integration
osv-scanner --sbom sbom-python.json --format json
```

### Continuous Monitoring

```yaml
# Scheduled vulnerability scanning
- name: SBOM Vulnerability Scan
  run: |
    # Generate fresh SBOM
    cyclonedx-py -o current-sbom.json
    
    # Scan for vulnerabilities
    grype sbom:current-sbom.json -o json > vuln-scan.json
    
    # Compare with baseline
    python scripts/compare_vulnerabilities.py baseline-sbom.json current-sbom.json
```

## SBOM Distribution

### Release Artifacts

Include SBOMs in every release:

```yaml
- name: Attach SBOM to Release
  uses: actions/upload-release-asset@v1
  with:
    upload_url: ${{ steps.create_release.outputs.upload_url }}
    asset_path: ./sbom-complete.json
    asset_name: agi-eval-sandbox-${{ github.ref }}-sbom.json
    asset_content_type: application/json
```

### Container Registry Integration

```dockerfile
# Add SBOM as container label
LABEL org.opencontainers.image.sbom="sbom-container.json"

# Include SBOM in image
COPY sbom-container.json /usr/share/doc/sbom.json
```

## Compliance Integration

### SLSA Level 2 Requirements

1. **Build Provenance**
   - Build environment documentation
   - Source integrity verification
   - Build process attestation

2. **SBOM Requirements**
   - Complete dependency listing
   - Build-time SBOM generation
   - Cryptographic attestation

### NIST SSDF Integration

Map SBOM components to NIST Secure Software Development Framework:

- **PO.1.1**: Define security requirements
- **PO.1.2**: Identify and document external components
- **PO.3.1**: Design software architecture
- **PS.2.1**: Use secure coding practices

## SBOM Verification

### Digital Signatures

```bash
# Sign SBOM with GPG
gpg --detach-sign --armor sbom-complete.json

# Verify SBOM signature
gpg --verify sbom-complete.json.asc sbom-complete.json
```

### Attestation Framework

```yaml
# Use SLSA attestations
- name: Generate SLSA Attestation
  uses: slsa-framework/slsa-github-generator@v1.2.0
  with:
    attestation-name: sbom-attestation
    sbom-path: sbom-complete.json
```

## Integration with Supply Chain Tools

### Dependency Track

```bash
# Upload SBOM to Dependency Track
curl -X POST \
  -H "X-API-Key: $DEPENDENCY_TRACK_API_KEY" \
  -H "Content-Type: multipart/form-data" \
  -F "project=$PROJECT_UUID" \
  -F "bom=@sbom-complete.json" \
  "$DEPENDENCY_TRACK_URL/api/v1/bom"
```

### FOSSA Integration

```yaml
- name: FOSSA Analysis
  uses: fossas/fossa-action@main
  with:
    api-key: ${{ secrets.FOSSA_API_KEY }}
    upload-sbom: true
    sbom-path: sbom-complete.json
```

## Best Practices

1. **Automate SBOM Generation**
   - Generate SBOMs in CI/CD pipeline
   - Update on every build/release
   - Version control SBOM templates

2. **Multi-Format Support**
   - Generate both SPDX and CycloneDX formats
   - Maintain format-specific tooling
   - Validate SBOM integrity

3. **Security Integration**
   - Scan SBOMs for vulnerabilities
   - Monitor for new security advisories
   - Integrate with threat intelligence

4. **Compliance Documentation**
   - Map to regulatory requirements
   - Maintain audit trails
   - Document exceptions and waivers

## Monitoring and Alerting

### SBOM Drift Detection

```python
# Monitor SBOM changes
def detect_sbom_drift(baseline_sbom, current_sbom):
    baseline_deps = set(baseline_sbom['components'])
    current_deps = set(current_sbom['components'])
    
    added = current_deps - baseline_deps
    removed = baseline_deps - current_deps
    
    if added or removed:
        send_alert(f"SBOM drift detected: +{len(added)} -{len(removed)}")
```

### Vulnerability Alerts

```bash
# Set up monitoring for new vulnerabilities
osv-scanner --sbom sbom-complete.json --format json | \
  jq '.results[].packages[].vulnerabilities[].id' | \
  xargs -I {} echo "New vulnerability detected: {}"
```