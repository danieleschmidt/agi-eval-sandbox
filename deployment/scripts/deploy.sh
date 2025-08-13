#!/bin/bash
set -euo pipefail

# AGI Evaluation Sandbox Deployment Script
# Supports Docker Compose and Kubernetes deployments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOYMENT_DIR="${PROJECT_ROOT}/deployment"

# Default values
ENVIRONMENT="${ENVIRONMENT:-production}"
DEPLOYMENT_TYPE="${DEPLOYMENT_TYPE:-docker}"
VERSION="${VERSION:-1.0.0}"
NAMESPACE="${NAMESPACE:-agi-eval}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Print banner
print_banner() {
    echo "==========================================================="
    echo "  🚀 AGI Evaluation Sandbox Deployment Script"
    echo "  Environment: ${ENVIRONMENT}"
    echo "  Deployment Type: ${DEPLOYMENT_TYPE}"
    echo "  Version: ${VERSION}"
    echo "==========================================================="
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if running as root (not recommended)
    if [[ $EUID -eq 0 ]]; then
        log_warning "Running as root is not recommended for security reasons"
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    # Check deployment-specific requirements
    case "${DEPLOYMENT_TYPE}" in
        "docker")
            if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
                log_error "Docker Compose is required but not installed"
                exit 1
            fi
            ;;
        "kubernetes"|"k8s")
            if ! command -v kubectl &> /dev/null; then
                log_error "kubectl is required but not installed"
                exit 1
            fi
            ;;
    esac
    
    log_success "Prerequisites check passed"
}

# Load environment variables
load_env_vars() {
    local env_file="${DEPLOYMENT_DIR}/.env.${ENVIRONMENT}"
    
    if [[ -f "${env_file}" ]]; then
        log_info "Loading environment variables from ${env_file}"
        set -a
        source "${env_file}"
        set +a
    else
        log_warning "Environment file ${env_file} not found, using defaults"
    fi
    
    # Set required variables with defaults
    export SECRET_KEY="${SECRET_KEY:-$(openssl rand -hex 32)}"
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"
    export REDIS_PASSWORD="${REDIS_PASSWORD:-$(openssl rand -base64 32)}"
    export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-admin123}"
    export BUILD_DATE="$(date -u +'%Y-%m-%dT%H:%M:%SZ')"
    export VCS_REF="$(git rev-parse HEAD 2>/dev/null || echo 'unknown')"
    
    log_success "Environment variables loaded"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "${PROJECT_ROOT}"
    
    # Build main application image
    docker build \
        -f deployment/docker/Dockerfile \
        --build-arg ENVIRONMENT="${ENVIRONMENT}" \
        --build-arg BUILD_DATE="${BUILD_DATE}" \
        --build-arg VERSION="${VERSION}" \
        --build-arg VCS_REF="${VCS_REF}" \
        -t "terragonlabs/agi-eval-sandbox:${VERSION}" \
        -t "terragonlabs/agi-eval-sandbox:latest" \
        .
    
    log_success "Docker images built successfully"
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "${DEPLOYMENT_DIR}"
    
    # Create necessary directories
    mkdir -p logs data cache
    
    # Pull latest images for external services
    docker-compose pull postgres redis nginx prometheus grafana jaeger
    
    # Deploy the stack
    docker-compose up -d --remove-orphans
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    for i in {1..60}; do
        if docker-compose ps | grep -q "Up (healthy)"; then
            break
        fi
        sleep 5
        echo -n "."
    done
    echo
    
    # Show service status
    docker-compose ps
    
    log_success "Docker deployment completed"
    log_info "Services available at:"
    log_info "  API: http://localhost:8080"
    log_info "  Grafana: http://localhost:3000 (admin/${GRAFANA_PASSWORD})"
    log_info "  Prometheus: http://localhost:9090"
    log_info "  Jaeger: http://localhost:16686"
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    # Create namespace
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -
    
    # Create secrets
    kubectl create secret generic agi-eval-secrets \
        --namespace="${NAMESPACE}" \
        --from-literal=database-url="postgresql://agi_eval:${POSTGRES_PASSWORD}@postgres:5432/agi_eval" \
        --from-literal=redis-url="redis://redis:6379/0" \
        --from-literal=secret-key="${SECRET_KEY}" \
        --from-literal=openai-api-key="${OPENAI_API_KEY:-}" \
        --from-literal=anthropic-api-key="${ANTHROPIC_API_KEY:-}" \
        --from-literal=google-api-key="${GOOGLE_API_KEY:-}" \
        --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply Kubernetes manifests
    kubectl apply -f "${DEPLOYMENT_DIR}/kubernetes/" --namespace="${NAMESPACE}"
    
    # Wait for rollout
    kubectl rollout status deployment/agi-eval-api --namespace="${NAMESPACE}" --timeout=600s
    
    # Show deployment status
    kubectl get pods,svc,ingress --namespace="${NAMESPACE}"
    
    log_success "Kubernetes deployment completed"
    
    # Get ingress URL
    local ingress_url
    ingress_url=$(kubectl get ingress agi-eval-ingress --namespace="${NAMESPACE}" -o jsonpath='{.spec.rules[0].host}' 2>/dev/null || echo "")
    if [[ -n "${ingress_url}" ]]; then
        log_info "API available at: https://${ingress_url}"
    fi
}

# Run health checks
run_health_checks() {
    log_info "Running health checks..."
    
    local health_url
    case "${DEPLOYMENT_TYPE}" in
        "docker")
            health_url="http://localhost:8080/health"
            ;;
        "kubernetes"|"k8s")
            # Port forward for health check
            kubectl port-forward service/agi-eval-api-service 8080:80 --namespace="${NAMESPACE}" &
            local port_forward_pid=$!
            sleep 5
            health_url="http://localhost:8080/health"
            ;;
    esac
    
    # Wait for API to be healthy
    for i in {1..30}; do
        if curl -f "${health_url}" &>/dev/null; then
            log_success "Health check passed"
            break
        fi
        sleep 10
        echo -n "."
    done
    echo
    
    # Cleanup port forward if used
    if [[ "${DEPLOYMENT_TYPE}" == "kubernetes" || "${DEPLOYMENT_TYPE}" == "k8s" ]]; then
        kill ${port_forward_pid} 2>/dev/null || true
    fi
}

# Generate deployment report
generate_report() {
    log_info "Generating deployment report..."
    
    local report_file="${DEPLOYMENT_DIR}/deployment-report-${ENVIRONMENT}-$(date +%Y%m%d-%H%M%S).json"
    
    cat > "${report_file}" << EOF
{
  "deployment": {
    "timestamp": "$(date -u +'%Y-%m-%dT%H:%M:%SZ')",
    "environment": "${ENVIRONMENT}",
    "deployment_type": "${DEPLOYMENT_TYPE}",
    "version": "${VERSION}",
    "vcs_ref": "${VCS_REF}",
    "namespace": "${NAMESPACE}"
  },
  "services": {
    "api": "agi-eval-api",
    "database": "postgres",
    "cache": "redis",
    "monitoring": ["prometheus", "grafana"],
    "tracing": "jaeger"
  },
  "endpoints": {
    "api": "http://localhost:8080",
    "grafana": "http://localhost:3000",
    "prometheus": "http://localhost:9090",
    "jaeger": "http://localhost:16686"
  }
}
EOF
    
    log_success "Deployment report saved to: ${report_file}"
}

# Show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Deploy AGI Evaluation Sandbox

OPTIONS:
    -e, --environment    Environment (production, staging, development) [default: production]
    -t, --type          Deployment type (docker, kubernetes, k8s) [default: docker]
    -v, --version       Version to deploy [default: 1.0.0]
    -n, --namespace     Kubernetes namespace [default: agi-eval]
    --build             Build images before deployment
    --health-check      Run health checks after deployment
    --report            Generate deployment report
    -h, --help          Show this help message

EXAMPLES:
    $0                                    # Deploy with defaults (Docker, production)
    $0 -t kubernetes -e staging          # Deploy to Kubernetes in staging
    $0 --build --health-check --report   # Full deployment with checks and report

ENVIRONMENT VARIABLES:
    SECRET_KEY          Application secret key
    POSTGRES_PASSWORD   PostgreSQL password
    REDIS_PASSWORD      Redis password
    OPENAI_API_KEY      OpenAI API key (optional)
    ANTHROPIC_API_KEY   Anthropic API key (optional)
    GOOGLE_API_KEY      Google API key (optional)

EOF
}

# Main function
main() {
    local build_images=false
    local run_health_checks=false
    local generate_report=false
    
    # Parse command line arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--type)
                DEPLOYMENT_TYPE="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            --build)
                build_images=true
                shift
                ;;
            --health-check)
                run_health_checks=true
                shift
                ;;
            --report)
                generate_report=true
                shift
                ;;
            -h|--help)
                show_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    # Validate deployment type
    case "${DEPLOYMENT_TYPE}" in
        "docker"|"kubernetes"|"k8s") ;;
        *)
            log_error "Invalid deployment type: ${DEPLOYMENT_TYPE}"
            exit 1
            ;;
    esac
    
    # Start deployment
    print_banner
    check_prerequisites
    load_env_vars
    
    if [[ "${build_images}" == true ]]; then
        build_images
    fi
    
    case "${DEPLOYMENT_TYPE}" in
        "docker")
            deploy_docker
            ;;
        "kubernetes"|"k8s")
            deploy_kubernetes
            ;;
    esac
    
    if [[ "${run_health_checks}" == true ]]; then
        run_health_checks
    fi
    
    if [[ "${generate_report}" == true ]]; then
        generate_report
    fi
    
    log_success "🎉 Deployment completed successfully!"
}

# Run main function with all arguments
main "$@"