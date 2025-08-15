#!/bin/bash

set -euo pipefail

# Production Deployment Script for AGI Evaluation Sandbox
# This script handles the complete production deployment process

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}
NAMESPACE="agi-eval-sandbox"
IMAGE_NAME="agi-eval-sandbox"
REGISTRY=${DOCKER_REGISTRY:-"your-registry.com"}

# Functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check required tools
    for tool in docker kubectl helm; do
        if ! command -v $tool &> /dev/null; then
            log_error "$tool is not installed"
            exit 1
        fi
    done
    
    # Check Kubernetes connection
    if ! kubectl cluster-info &> /dev/null; then
        log_error "Cannot connect to Kubernetes cluster"
        exit 1
    fi
    
    # Check Docker daemon
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

run_quality_gates() {
    log_info "Running quality gates..."
    
    # Run comprehensive tests
    if ! python3 test_autonomous_implementation.py; then
        log_error "Quality gates failed"
        exit 1
    fi
    
    log_success "Quality gates passed"
}

build_and_push_image() {
    log_info "Building and pushing Docker image..."
    
    # Build optimized image
    docker build -f Dockerfile.optimized -t ${IMAGE_NAME}:${VERSION} .
    docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    docker tag ${IMAGE_NAME}:${VERSION} ${REGISTRY}/${IMAGE_NAME}:latest
    
    # Security scan
    log_info "Running security scan on Docker image..."
    if command -v trivy &> /dev/null; then
        trivy image --severity HIGH,CRITICAL ${IMAGE_NAME}:${VERSION}
    else
        log_warning "Trivy not found, skipping security scan"
    fi
    
    # Push to registry
    docker push ${REGISTRY}/${IMAGE_NAME}:${VERSION}
    docker push ${REGISTRY}/${IMAGE_NAME}:latest
    
    log_success "Image built and pushed: ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
}

deploy_infrastructure() {
    log_info "Deploying infrastructure components..."
    
    # Create namespace
    kubectl apply -f deployment/kubernetes/production/namespace.yaml
    
    # Deploy Redis
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo update
    
    helm upgrade --install redis bitnami/redis \
        --namespace ${NAMESPACE} \
        --set auth.enabled=false \
        --set architecture=standalone \
        --set master.persistence.enabled=true \
        --set master.persistence.size=10Gi \
        --set master.resources.requests.memory=512Mi \
        --set master.resources.requests.cpu=250m \
        --set master.resources.limits.memory=1Gi \
        --set master.resources.limits.cpu=500m
    
    # Deploy PostgreSQL
    helm upgrade --install postgres bitnami/postgresql \
        --namespace ${NAMESPACE} \
        --set auth.postgresPassword=${POSTGRES_PASSWORD} \
        --set auth.database=agi_eval_sandbox \
        --set primary.persistence.enabled=true \
        --set primary.persistence.size=20Gi \
        --set primary.resources.requests.memory=1Gi \
        --set primary.resources.requests.cpu=500m \
        --set primary.resources.limits.memory=2Gi \
        --set primary.resources.limits.cpu=1000m
    
    # Wait for infrastructure to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=redis -n ${NAMESPACE} --timeout=300s
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/name=postgresql -n ${NAMESPACE} --timeout=300s
    
    log_success "Infrastructure deployed successfully"
}

deploy_monitoring() {
    log_info "Deploying monitoring stack..."
    
    # Add monitoring Helm repositories
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo add jaegertracing https://jaegertracing.github.io/helm-charts
    helm repo update
    
    # Deploy Prometheus
    helm upgrade --install prometheus prometheus-community/kube-prometheus-stack \
        --namespace monitoring \
        --create-namespace \
        --set prometheus.prometheusSpec.retention=30d \
        --set prometheus.prometheusSpec.storageSpec.volumeClaimTemplate.spec.resources.requests.storage=50Gi \
        --set grafana.adminPassword=${GRAFANA_PASSWORD}
    
    # Deploy Jaeger
    helm upgrade --install jaeger jaegertracing/jaeger \
        --namespace monitoring \
        --set provisionDataStore.cassandra=false \
        --set storage.type=memory \
        --set allInOne.enabled=true
    
    log_success "Monitoring stack deployed"
}

deploy_application() {
    log_info "Deploying application..."
    
    # Create secrets
    kubectl create secret generic postgres-secret \
        --from-literal=database-url="postgresql://postgres:${POSTGRES_PASSWORD}@postgres-postgresql:5432/agi_eval_sandbox" \
        --namespace ${NAMESPACE} \
        --dry-run=client -o yaml | kubectl apply -f -
    
    if [ -n "${SENTRY_DSN:-}" ]; then
        kubectl create secret generic sentry-secret \
            --from-literal=dsn="${SENTRY_DSN}" \
            --namespace ${NAMESPACE} \
            --dry-run=client -o yaml | kubectl apply -f -
    fi
    
    # Update image in deployment
    sed "s|agi-eval-sandbox:1.0.0|${REGISTRY}/${IMAGE_NAME}:${VERSION}|g" \
        deployment/kubernetes/production/deployment.yaml | kubectl apply -f -
    
    # Wait for deployment
    kubectl rollout status deployment/agi-eval-sandbox-api -n ${NAMESPACE} --timeout=600s
    kubectl rollout status deployment/agi-eval-sandbox-worker -n ${NAMESPACE} --timeout=600s
    
    log_success "Application deployed successfully"
}

setup_ingress() {
    log_info "Setting up ingress..."
    
    # Deploy NGINX ingress controller if not exists
    if ! kubectl get ingressclass nginx &> /dev/null; then
        helm upgrade --install ingress-nginx ingress-nginx/ingress-nginx \
            --namespace ingress-nginx \
            --create-namespace \
            --set controller.replicaCount=2 \
            --set controller.nodeSelector."kubernetes\.io/os"=linux \
            --set defaultBackend.nodeSelector."kubernetes\.io/os"=linux
    fi
    
    # Apply ingress configuration
    cat <<EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: agi-eval-sandbox-ingress
  namespace: ${NAMESPACE}
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
    nginx.ingress.kubernetes.io/rate-limit: "100"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  ingressClassName: nginx
  tls:
  - hosts:
    - api.agi-eval-sandbox.com
    secretName: agi-eval-sandbox-tls
  rules:
  - host: api.agi-eval-sandbox.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: agi-eval-sandbox-service
            port:
              number: 80
EOF
    
    log_success "Ingress configured"
}

run_smoke_tests() {
    log_info "Running smoke tests..."
    
    # Wait for service to be ready
    kubectl wait --for=condition=ready pod -l app.kubernetes.io/component=api -n ${NAMESPACE} --timeout=300s
    
    # Port forward for testing
    kubectl port-forward svc/agi-eval-sandbox-service 8080:80 -n ${NAMESPACE} &
    PORT_FORWARD_PID=$!
    
    sleep 10
    
    # Health check
    if curl -f http://localhost:8080/health; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        kill $PORT_FORWARD_PID 2>/dev/null || true
        exit 1
    fi
    
    # API test
    if curl -f http://localhost:8080/api/v1; then
        log_success "API test passed"
    else
        log_error "API test failed"
        kill $PORT_FORWARD_PID 2>/dev/null || true
        exit 1
    fi
    
    kill $PORT_FORWARD_PID 2>/dev/null || true
    
    log_success "Smoke tests passed"
}

print_deployment_info() {
    log_info "Deployment Information:"
    echo "----------------------------------------"
    echo "Environment: ${ENVIRONMENT}"
    echo "Version: ${VERSION}"
    echo "Namespace: ${NAMESPACE}"
    echo "Image: ${REGISTRY}/${IMAGE_NAME}:${VERSION}"
    echo ""
    echo "Kubernetes Resources:"
    kubectl get all -n ${NAMESPACE}
    echo ""
    echo "Ingress:"
    kubectl get ingress -n ${NAMESPACE}
    echo ""
    echo "Monitoring:"
    echo "- Prometheus: http://prometheus.monitoring.local"
    echo "- Grafana: http://grafana.monitoring.local (admin/${GRAFANA_PASSWORD})"
    echo "- Jaeger: http://jaeger.monitoring.local"
    echo "----------------------------------------"
}

cleanup_on_failure() {
    log_error "Deployment failed. Cleaning up..."
    kubectl delete namespace ${NAMESPACE} --ignore-not-found=true
    log_info "Cleanup completed"
}

# Main deployment process
main() {
    log_info "Starting production deployment for AGI Evaluation Sandbox"
    log_info "Environment: ${ENVIRONMENT}, Version: ${VERSION}"
    
    # Set trap for cleanup on failure
    trap cleanup_on_failure ERR
    
    # Check environment variables
    if [ -z "${POSTGRES_PASSWORD:-}" ]; then
        log_error "POSTGRES_PASSWORD environment variable is required"
        exit 1
    fi
    
    if [ -z "${GRAFANA_PASSWORD:-}" ]; then
        log_error "GRAFANA_PASSWORD environment variable is required"
        exit 1
    fi
    
    # Run deployment steps
    check_prerequisites
    run_quality_gates
    build_and_push_image
    deploy_infrastructure
    deploy_monitoring
    deploy_application
    setup_ingress
    run_smoke_tests
    
    print_deployment_info
    
    log_success "🚀 Production deployment completed successfully!"
    log_info "Your AGI Evaluation Sandbox is now running in production."
}

# Run main function
main "$@"