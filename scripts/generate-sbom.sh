#!/bin/bash

# Software Bill of Materials (SBOM) Generation Script
# AGI Evaluation Sandbox

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Configuration
OUTPUT_DIR="sbom"
FORMATS=("spdx-json" "cyclone-json" "cyclone-xml")
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies for SBOM generation..."
    
    # Check for syft (for filesystem scanning)
    if ! command -v syft >/dev/null 2>&1; then
        log_warning "syft not found. Installing..."
        curl -sSfL https://raw.githubusercontent.com/anchore/syft/main/install.sh | sh -s -- -b /usr/local/bin
    fi
    
    # Check for cyclone-dx tools
    if ! command -v cyclonedx-py >/dev/null 2>&1; then
        log_warning "cyclonedx-py not found. Installing via pip..."
        pip install cyclonedx-bom[requirements]
    fi
    
    log_success "Dependencies checked"
}

# Generate Python SBOM
generate_python_sbom() {
    log_info "Generating Python SBOM..."
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR/python"
    
    # Generate from requirements/pyproject.toml
    if [[ -f "pyproject.toml" ]]; then
        log_info "Generating SBOM from pyproject.toml..."
        cyclonedx-py -o "$OUTPUT_DIR/python/python-cyclonedx.json" pyproject.toml
    elif [[ -f "requirements.txt" ]]; then
        log_info "Generating SBOM from requirements.txt..."
        cyclonedx-py -r -o "$OUTPUT_DIR/python/python-cyclonedx.json" requirements.txt
    else
        log_warning "No Python requirements file found"
    fi
    
    # Generate using pip freeze for current environment
    if command -v pip >/dev/null 2>&1; then
        log_info "Generating SBOM from current pip environment..."
        pip freeze > "$OUTPUT_DIR/python/requirements-frozen.txt"
        cyclonedx-py -r -o "$OUTPUT_DIR/python/python-env-cyclonedx.json" "$OUTPUT_DIR/python/requirements-frozen.txt"
    fi
}

# Generate Node.js SBOM
generate_nodejs_sbom() {
    log_info "Generating Node.js SBOM..."
    
    if [[ ! -f "package.json" ]]; then
        log_warning "No package.json found, skipping Node.js SBOM"
        return
    fi
    
    mkdir -p "$OUTPUT_DIR/nodejs"
    
    # Generate using npm list
    if command -v npm >/dev/null 2>&1; then
        log_info "Generating Node.js dependency tree..."
        npm list --json > "$OUTPUT_DIR/nodejs/npm-dependencies.json" 2>/dev/null || true
        npm list --production --json > "$OUTPUT_DIR/nodejs/npm-prod-dependencies.json" 2>/dev/null || true
    fi
    
    # Generate CycloneDX SBOM for Node.js if cyclonedx-node is available
    if command -v cyclonedx-node >/dev/null 2>&1; then
        log_info "Generating CycloneDX SBOM for Node.js..."
        cyclonedx-node -o "$OUTPUT_DIR/nodejs/nodejs-cyclonedx.json"
    fi
}

# Generate filesystem SBOM using syft
generate_filesystem_sbom() {
    log_info "Generating filesystem SBOM using syft..."
    
    mkdir -p "$OUTPUT_DIR/filesystem"
    
    # Scan current directory
    syft . -o spdx-json="$OUTPUT_DIR/filesystem/filesystem-spdx.json"
    syft . -o cyclonedx-json="$OUTPUT_DIR/filesystem/filesystem-cyclonedx.json"
    syft . -o cyclonedx-xml="$OUTPUT_DIR/filesystem/filesystem-cyclonedx.xml"
    syft . -o table="$OUTPUT_DIR/filesystem/filesystem-table.txt"
}

# Generate Docker image SBOM
generate_docker_sbom() {
    log_info "Generating Docker image SBOM..."
    
    # Check if Docker image exists
    IMAGE_NAME="agi-eval-sandbox:latest"
    if ! docker image inspect "$IMAGE_NAME" >/dev/null 2>&1; then
        log_warning "Docker image $IMAGE_NAME not found, building..."
        docker build -t "$IMAGE_NAME" . || {
            log_warning "Failed to build Docker image, skipping Docker SBOM"
            return
        }
    fi
    
    mkdir -p "$OUTPUT_DIR/docker"
    
    # Generate SBOM for Docker image
    syft "$IMAGE_NAME" -o spdx-json="$OUTPUT_DIR/docker/docker-spdx.json"
    syft "$IMAGE_NAME" -o cyclonedx-json="$OUTPUT_DIR/docker/docker-cyclonedx.json"
    syft "$IMAGE_NAME" -o cyclonedx-xml="$OUTPUT_DIR/docker/docker-cyclonedx.xml"
    syft "$IMAGE_NAME" -o table="$OUTPUT_DIR/docker/docker-table.txt"
}

# Generate comprehensive SBOM metadata
generate_metadata() {
    log_info "Generating SBOM metadata..."
    
    cat > "$OUTPUT_DIR/metadata.json" <<EOF
{
  "name": "agi-eval-sandbox",
  "version": "$(git describe --tags --always 2>/dev/null || echo 'unknown')",
  "description": "One-click evaluation environment bundling DeepEval, HELM-Lite, MT-Bench, and custom benchmarks",
  "repository": "$(git config --get remote.origin.url 2>/dev/null || echo 'unknown')",
  "branch": "$(git branch --show-current 2>/dev/null || echo 'unknown')",
  "commit": "$(git rev-parse HEAD 2>/dev/null || echo 'unknown')",
  "timestamp": "$TIMESTAMP",
  "generator": "custom-sbom-generator",
  "generator_version": "1.0.0",
  "components": {
    "python": $(find "$OUTPUT_DIR/python" -name "*.json" -type f 2>/dev/null | wc -l || echo 0),
    "nodejs": $(find "$OUTPUT_DIR/nodejs" -name "*.json" -type f 2>/dev/null | wc -l || echo 0),
    "filesystem": $(find "$OUTPUT_DIR/filesystem" -name "*.json" -type f 2>/dev/null | wc -l || echo 0),
    "docker": $(find "$OUTPUT_DIR/docker" -name "*.json" -type f 2>/dev/null | wc -l || echo 0)
  },
  "formats": ["spdx-json", "cyclonedx-json", "cyclonedx-xml", "table"],
  "scan_paths": ["."],
  "excluded_paths": [
    ".git",
    "node_modules", 
    "__pycache__",
    ".pytest_cache",
    "venv",
    ".venv",
    "dist",
    "build"
  ]
}
EOF
}

# Generate security analysis
generate_security_analysis() {
    log_info "Generating security analysis..."
    
    mkdir -p "$OUTPUT_DIR/security"
    
    # Python security analysis
    if command -v safety >/dev/null 2>&1; then
        log_info "Running Python security scan with safety..."
        safety check --json --output "$OUTPUT_DIR/security/python-vulns.json" || true
        safety check --output "$OUTPUT_DIR/security/python-vulns.txt" || true
    fi
    
    # Node.js security analysis  
    if command -v npm >/dev/null 2>&1 && [[ -f "package.json" ]]; then
        log_info "Running Node.js security audit..."
        npm audit --json > "$OUTPUT_DIR/security/nodejs-audit.json" 2>/dev/null || true
        npm audit > "$OUTPUT_DIR/security/nodejs-audit.txt" 2>/dev/null || true
    fi
    
    # Bandit security analysis for Python
    if command -v bandit >/dev/null 2>&1; then
        log_info "Running bandit security analysis..."
        bandit -r src/ -f json -o "$OUTPUT_DIR/security/bandit-results.json" || true
        bandit -r src/ -o "$OUTPUT_DIR/security/bandit-results.txt" || true
    fi
}

# Generate compliance report
generate_compliance_report() {
    log_info "Generating compliance report..."
    
    cat > "$OUTPUT_DIR/compliance-report.md" <<EOF
# Software Bill of Materials (SBOM) Compliance Report
## AGI Evaluation Sandbox

**Generated:** $TIMESTAMP  
**Version:** $(git describe --tags --always 2>/dev/null || echo 'unknown')  
**Commit:** $(git rev-parse HEAD 2>/dev/null || echo 'unknown')  

## Executive Summary

This SBOM was generated to provide transparency into the software supply chain of the AGI Evaluation Sandbox project. It includes all dependencies, both direct and transitive, across multiple ecosystems.

## Components Analyzed

- **Python Dependencies:** Analyzed via pip/pyproject.toml
- **Node.js Dependencies:** Analyzed via npm/package.json  
- **Filesystem Scan:** Complete filesystem analysis via syft
- **Container Images:** Docker image component analysis
- **Security Vulnerabilities:** Known vulnerability analysis

## Compliance Standards

This SBOM is generated in compliance with:
- **SPDX 2.3:** Software Package Data Exchange format
- **CycloneDX 1.4:** OWASP CycloneDX specification
- **NTIA Minimum Elements:** As defined by NTIA SBOM requirements

## Files Generated

### SPDX Format
- \`filesystem/filesystem-spdx.json\` - Complete filesystem SPDX SBOM
- \`docker/docker-spdx.json\` - Docker image SPDX SBOM

### CycloneDX Format  
- \`python/python-cyclonedx.json\` - Python dependencies CycloneDX SBOM
- \`nodejs/nodejs-cyclonedx.json\` - Node.js dependencies CycloneDX SBOM
- \`filesystem/filesystem-cyclonedx.json\` - Filesystem CycloneDX SBOM
- \`docker/docker-cyclonedx.json\` - Docker image CycloneDX SBOM

### Security Analysis
- \`security/python-vulns.json\` - Python vulnerability analysis
- \`security/nodejs-audit.json\` - Node.js security audit
- \`security/bandit-results.json\` - Python static security analysis

### Human Readable
- \`filesystem/filesystem-table.txt\` - Human-readable component list
- \`docker/docker-table.txt\` - Human-readable Docker component list

## Usage

These SBOM files can be used for:
- Supply chain security analysis
- Vulnerability management
- Compliance reporting
- License compliance verification
- Dependency tracking and management

## Verification

To verify the integrity of generated SBOMs:
1. Check file checksums against \`checksums.txt\`
2. Validate SPDX files using SPDX validation tools
3. Validate CycloneDX files using CycloneDX validation tools

## Contact

For questions about this SBOM or the software supply chain, contact:
- Repository: $(git config --get remote.origin.url 2>/dev/null || echo 'unknown')
- Issues: GitHub Issues tracker
EOF
}

# Generate checksums
generate_checksums() {
    log_info "Generating file checksums..."
    
    find "$OUTPUT_DIR" -type f -name "*.json" -o -name "*.xml" -o -name "*.txt" | \
    sort | \
    xargs sha256sum > "$OUTPUT_DIR/checksums.txt"
}

# Create archive
create_archive() {
    log_info "Creating SBOM archive..."
    
    tar -czf "sbom-$(date +%Y%m%d-%H%M%S).tar.gz" "$OUTPUT_DIR"
    
    log_success "SBOM archive created: sbom-$(date +%Y%m%d-%H%M%S).tar.gz"
}

# Display summary
show_summary() {
    log_info "SBOM Generation Summary"
    echo "======================="
    echo "Output directory: $OUTPUT_DIR"
    echo "Files generated:"
    find "$OUTPUT_DIR" -type f | sort | sed 's/^/  - /'
    echo ""
    echo "File sizes:"
    find "$OUTPUT_DIR" -type f -exec ls -lh {} \; | awk '{print "  - " $9 ": " $5}'
    echo ""
    log_success "SBOM generation completed successfully!"
}

# Main execution
main() {
    log_info "Starting SBOM generation for AGI Evaluation Sandbox"
    
    # Create output directory
    mkdir -p "$OUTPUT_DIR"
    
    # Check and install dependencies
    check_dependencies
    
    # Generate different types of SBOMs
    generate_python_sbom
    generate_nodejs_sbom
    generate_filesystem_sbom
    generate_docker_sbom
    
    # Generate metadata and reports
    generate_metadata
    generate_security_analysis
    generate_compliance_report
    generate_checksums
    
    # Create archive
    create_archive
    
    # Show summary
    show_summary
}

# Run main function
main "$@"