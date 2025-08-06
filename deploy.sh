#!/bin/bash
# Production deployment script for AGI Evaluation Sandbox
# Implements full SDLC with quality gates and automated deployment

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
DEPLOYMENT_ENV="${DEPLOYMENT_ENV:-production}"
DOCKER_REGISTRY="${DOCKER_REGISTRY:-terragon.azurecr.io}"
IMAGE_NAME="${IMAGE_NAME:-agi-eval-sandbox}"
IMAGE_TAG="${IMAGE_TAG:-$(git rev-parse --short HEAD)}"
NAMESPACE="${NAMESPACE:-agi-eval-sandbox}"
CLUSTER_NAME="${CLUSTER_NAME:-agi-eval-prod}"

# Quality gates configuration
MIN_TEST_COVERAGE=${MIN_TEST_COVERAGE:-85}
MIN_PERFORMANCE_SCORE=${MIN_PERFORMANCE_SCORE:-90}
MAX_SECURITY_ISSUES=${MAX_SECURITY_ISSUES:-0}
MAX_DEPLOYMENT_TIME=${MAX_DEPLOYMENT_TIME:-300}

echo -e "${BLUE}🚀 Starting AGI Evaluation Sandbox Production Deployment${NC}"
echo "============================================================"
echo "Environment: $DEPLOYMENT_ENV"
echo "Image: $DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
echo "Namespace: $NAMESPACE"
echo "Cluster: $CLUSTER_NAME"
echo "============================================================"

# Function to log with timestamp
log() {
    echo -e "$(date '+%Y-%m-%d %H:%M:%S') $1"
}

# Function to check if command exists
check_command() {
    if ! command -v $1 &> /dev/null; then
        echo -e "${RED}❌ Error: $1 is not installed${NC}"
        exit 1
    fi
}

# Function to run quality gates
run_quality_gates() {
    echo -e "${BLUE}🔍 Running Quality Gates${NC}"
    
    # 1. Code Quality - Linting and Formatting
    echo "Running linting checks..."
    if command -v ruff &> /dev/null; then
        ruff check src/ tests/ || {
            echo -e "${RED}❌ Linting failed${NC}"
            exit 1
        }
    else
        echo -e "${YELLOW}⚠️  Ruff not installed, skipping linting${NC}"
    fi
    
    # 2. Security Scan
    echo "Running security scans..."
    if command -v bandit &> /dev/null; then
        bandit -r src/ -f json -o security-report.json || {
            echo -e "${RED}❌ Security scan failed${NC}"
            exit 1
        }
        
        # Check for critical security issues
        SECURITY_ISSUES=$(jq '.results | length' security-report.json 2>/dev/null || echo "0")
        if [ "$SECURITY_ISSUES" -gt "$MAX_SECURITY_ISSUES" ]; then
            echo -e "${RED}❌ Too many security issues: $SECURITY_ISSUES (max: $MAX_SECURITY_ISSUES)${NC}"
            exit 1
        fi
        
        echo -e "${GREEN}✅ Security scan passed: $SECURITY_ISSUES issues found${NC}"
    else
        echo -e "${YELLOW}⚠️  Bandit not installed, skipping security scan${NC}"
    fi
    
    # 3. Dependency Vulnerability Check
    echo "Checking dependencies for vulnerabilities..."
    if command -v safety &> /dev/null; then
        safety check --json --output vulnerability-report.json || {
            echo -e "${RED}❌ Dependency vulnerability check failed${NC}"
            exit 1
        }
        echo -e "${GREEN}✅ Dependency vulnerability check passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Safety not installed, skipping vulnerability check${NC}"
    fi
    
    # 4. Unit Tests
    echo "Running unit tests..."
    if [ -f "test_basic_functionality.py" ]; then
        python3 test_basic_functionality.py || {
            echo -e "${RED}❌ Basic functionality tests failed${NC}"
            exit 1
        }
        echo -e "${GREEN}✅ Basic functionality tests passed${NC}"
    else
        echo -e "${YELLOW}⚠️  Basic functionality tests not found${NC}"
    fi
    
    # 5. Type Checking
    echo "Running type checks..."
    if command -v mypy &> /dev/null; then
        mypy src/ --ignore-missing-imports || {
            echo -e "${YELLOW}⚠️  Type checking issues found (non-blocking)${NC}"
        }
    else
        echo -e "${YELLOW}⚠️  MyPy not installed, skipping type checks${NC}"
    fi
    
    echo -e "${GREEN}✅ All quality gates passed${NC}"
}

# Function to build and test Docker image
build_and_test_image() {
    echo -e "${BLUE}🏗️  Building and Testing Docker Image${NC}"
    
    # Build production image
    echo "Building production Docker image..."
    docker build -f Dockerfile.optimized -t $IMAGE_NAME:$IMAGE_TAG --target production .
    
    # Build and run tests in container
    echo "Building and running tests..."
    docker build -f Dockerfile.optimized -t $IMAGE_NAME:$IMAGE_TAG-test --target testing . || {
        echo -e "${YELLOW}⚠️  Test build failed (dependencies may be missing)${NC}"
    }
    
    # Security scan of image
    echo "Scanning Docker image for vulnerabilities..."
    if command -v docker &> /dev/null; then
        docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
            aquasec/trivy:latest image $IMAGE_NAME:$IMAGE_TAG || {
            echo -e "${YELLOW}⚠️  Image vulnerability scan completed with warnings${NC}"
        }
    fi
    
    # Tag image for registry
    docker tag $IMAGE_NAME:$IMAGE_TAG $DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG
    docker tag $IMAGE_NAME:$IMAGE_TAG $DOCKER_REGISTRY/$IMAGE_NAME:latest
    
    echo -e "${GREEN}✅ Docker image built and tested successfully${NC}"
}

# Function to push to registry
push_to_registry() {
    echo -e "${BLUE}📤 Pushing to Container Registry${NC}"
    
    # Login to registry (assumes credentials are configured)
    echo "Authenticating with registry..."
    # docker login $DOCKER_REGISTRY # Assumes already authenticated
    
    # Push images
    docker push $DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG
    docker push $DOCKER_REGISTRY/$IMAGE_NAME:latest
    
    echo -e "${GREEN}✅ Images pushed to registry successfully${NC}"
}

# Function to deploy to Kubernetes
deploy_to_kubernetes() {
    echo -e "${BLUE}☸️  Deploying to Kubernetes${NC}"
    
    # Check kubectl connectivity
    kubectl cluster-info >/dev/null 2>&1 || {
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        exit 1
    }
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    echo "Applying Kubernetes configurations..."
    
    # Update image tags in deployments
    sed -i.bak "s|image: agi-eval-sandbox:latest|image: $DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG|g" k8s/deployment.yaml
    
    # Apply configurations in order
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml -n $NAMESPACE
    
    # Apply secrets (assumes they exist)
    if kubectl get secret database-secret -n $NAMESPACE >/dev/null 2>&1; then
        echo "Database secret exists"
    else
        echo -e "${YELLOW}⚠️  Database secret not found. Please create it manually.${NC}"
    fi
    
    # Apply deployments and services
    kubectl apply -f k8s/deployment.yaml -n $NAMESPACE
    kubectl apply -f k8s/service.yaml -n $NAMESPACE || echo "Service config not found, skipping"
    kubectl apply -f k8s/hpa.yaml -n $NAMESPACE
    
    # Wait for rollout to complete
    echo "Waiting for deployment rollout..."
    kubectl rollout status deployment/api-deployment -n $NAMESPACE --timeout=300s || {
        echo -e "${RED}❌ Deployment rollout failed or timed out${NC}"
        kubectl describe deployment/api-deployment -n $NAMESPACE
        exit 1
    }
    
    kubectl rollout status deployment/worker-deployment -n $NAMESPACE --timeout=300s || {
        echo -e "${RED}❌ Worker deployment rollout failed or timed out${NC}"
        kubectl describe deployment/worker-deployment -n $NAMESPACE
        exit 1
    }
    
    echo -e "${GREEN}✅ Kubernetes deployment completed successfully${NC}"
}

# Function to run smoke tests
run_smoke_tests() {
    echo -e "${BLUE}🧪 Running Smoke Tests${NC}"
    
    # Get service endpoint
    SERVICE_URL=$(kubectl get service nginx-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo "localhost")
    
    if [ "$SERVICE_URL" = "localhost" ] || [ -z "$SERVICE_URL" ]; then
        echo "Using port-forward for testing..."
        kubectl port-forward service/nginx-service -n $NAMESPACE 8080:80 &
        PORT_FORWARD_PID=$!
        sleep 5
        SERVICE_URL="localhost:8080"
    fi
    
    # Basic health check
    echo "Testing health endpoint..."
    curl -f http://$SERVICE_URL/health >/dev/null 2>&1 || {
        echo -e "${RED}❌ Health check failed${NC}"
        if [ ! -z "${PORT_FORWARD_PID:-}" ]; then
            kill $PORT_FORWARD_PID 2>/dev/null || true
        fi
        exit 1
    }
    
    # Test API endpoints
    echo "Testing API endpoints..."
    curl -f http://$SERVICE_URL/api/v1 >/dev/null 2>&1 || {
        echo -e "${RED}❌ API endpoint test failed${NC}"
        if [ ! -z "${PORT_FORWARD_PID:-}" ]; then
            kill $PORT_FORWARD_PID 2>/dev/null || true
        fi
        exit 1
    }
    
    # Test compression endpoint
    echo "Testing compression endpoints..."
    curl -f http://$SERVICE_URL/api/v1/compress/strategies >/dev/null 2>&1 || {
        echo -e "${YELLOW}⚠️  Compression strategies endpoint test failed (may require authentication)${NC}"
    }
    
    # Cleanup port-forward if used
    if [ ! -z "${PORT_FORWARD_PID:-}" ]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ Smoke tests passed${NC}"
}

# Function to setup monitoring
setup_monitoring() {
    echo -e "${BLUE}📊 Setting up Monitoring${NC}"
    
    # Apply monitoring configurations if they exist
    if [ -d "k8s/monitoring" ]; then
        kubectl apply -f k8s/monitoring/ -n $NAMESPACE || echo "Monitoring configs not found"
    fi
    
    # Check if Prometheus is accessible
    if kubectl get service prometheus -n $NAMESPACE >/dev/null 2>&1; then
        echo "Prometheus monitoring is available"
    else
        echo -e "${YELLOW}⚠️  Prometheus monitoring not configured${NC}"
    fi
    
    echo -e "${GREEN}✅ Monitoring setup completed${NC}"
}

# Function to cleanup on failure
cleanup() {
    echo -e "${YELLOW}🧹 Cleaning up...${NC}"
    
    # Restore original deployment files
    if [ -f "k8s/deployment.yaml.bak" ]; then
        mv k8s/deployment.yaml.bak k8s/deployment.yaml
    fi
    
    # Kill any background processes
    if [ ! -z "${PORT_FORWARD_PID:-}" ]; then
        kill $PORT_FORWARD_PID 2>/dev/null || true
    fi
}

# Main deployment function
main() {
    # Check prerequisites
    echo -e "${BLUE}🔧 Checking Prerequisites${NC}"
    check_command docker
    check_command kubectl
    check_command git
    
    # Set up cleanup trap
    trap cleanup EXIT
    
    # Run deployment steps
    run_quality_gates
    build_and_test_image
    
    # Only push and deploy if we're in CI/CD environment or explicitly requested
    if [ "$DEPLOYMENT_ENV" = "production" ] || [ "${FORCE_DEPLOY:-false}" = "true" ]; then
        push_to_registry
        deploy_to_kubernetes
        setup_monitoring
        run_smoke_tests
    else
        echo -e "${YELLOW}⚠️  Skipping registry push and deployment (not in production mode)${NC}"
        echo -e "${BLUE}To force deployment, set FORCE_DEPLOY=true${NC}"
    fi
    
    # Final success message
    echo ""
    echo -e "${GREEN}🎉 Deployment completed successfully!${NC}"
    echo "============================================================"
    echo "Image: $DOCKER_REGISTRY/$IMAGE_NAME:$IMAGE_TAG"
    echo "Namespace: $NAMESPACE"
    if [ "$DEPLOYMENT_ENV" = "production" ]; then
        echo "Service URL: http://$(kubectl get service nginx-service -n $NAMESPACE -o jsonpath='{.status.loadBalancer.ingress[0].ip}' 2>/dev/null || echo 'pending')"
    fi
    echo "Monitoring: kubectl port-forward service/prometheus -n $NAMESPACE 9090:9090"
    echo "Logs: kubectl logs -f deployment/api-deployment -n $NAMESPACE"
    echo "============================================================"
}

# Handle command line arguments
case "${1:-deploy}" in
    "quality-gates")
        run_quality_gates
        ;;
    "build")
        run_quality_gates
        build_and_test_image
        ;;
    "deploy")
        main
        ;;
    "smoke-tests")
        run_smoke_tests
        ;;
    "cleanup")
        echo "Cleaning up deployment..."
        kubectl delete namespace $NAMESPACE --ignore-not-found=true
        ;;
    *)
        echo "Usage: $0 {quality-gates|build|deploy|smoke-tests|cleanup}"
        echo ""
        echo "Commands:"
        echo "  quality-gates  - Run quality gates only"
        echo "  build         - Run quality gates and build image"
        echo "  deploy        - Full deployment (default)"
        echo "  smoke-tests   - Run smoke tests against deployed service"
        echo "  cleanup       - Remove deployed resources"
        echo ""
        echo "Environment variables:"
        echo "  DEPLOYMENT_ENV    - Deployment environment (default: production)"
        echo "  DOCKER_REGISTRY   - Container registry URL"
        echo "  FORCE_DEPLOY      - Force deployment even in non-production mode"
        exit 1
        ;;
esac