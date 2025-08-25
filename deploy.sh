#!/bin/bash
set -e

echo "🚀 Starting AGI Evaluation Sandbox Production Deployment"

# Configuration
ENVIRONMENT=${1:-production}
VERSION=${2:-latest}

echo "Environment: $ENVIRONMENT"
echo "Version: $VERSION"

# Pre-deployment checks
echo "📋 Running pre-deployment checks..."
python3 comprehensive_quality_gates_test.py
if [ $? -ne 0 ]; then
    echo "❌ Quality gates failed. Aborting deployment."
    exit 1
fi

# Build and test Docker image
echo "🏗️  Building Docker image..."
docker build -t agi-eval-sandbox:$VERSION .

# Run container tests
echo "🧪 Running container tests..."
docker run --rm agi-eval-sandbox:$VERSION python3 -m pytest tests/ -v

# Deploy based on environment
if [ "$ENVIRONMENT" = "production" ]; then
    echo "🌍 Deploying to production..."
    docker-compose -f docker-compose.prod.yml down
    docker-compose -f docker-compose.prod.yml up -d --force-recreate
elif [ "$ENVIRONMENT" = "staging" ]; then
    echo "🎭 Deploying to staging..."
    docker-compose -f docker-compose.staging.yml down
    docker-compose -f docker-compose.staging.yml up -d --force-recreate
else
    echo "🛠️  Deploying to development..."
    docker-compose down
    docker-compose up -d --build
fi

# Health check
echo "🏥 Performing health checks..."
sleep 30

# Check API health
for i in {1..10}; do
    if curl -f http://localhost:8080/health; then
        echo "✅ API health check passed"
        break
    else
        echo "⏳ Waiting for API to be healthy... ($i/10)"
        sleep 10
    fi
done

# Check dashboard
if curl -f http://localhost:3000; then
    echo "✅ Dashboard health check passed"
else
    echo "⚠️  Dashboard may not be fully ready"
fi

echo "🎉 Deployment completed successfully!"
echo "📊 API: http://localhost:8080"
echo "🖥️  Dashboard: http://localhost:3000"
echo "📈 Monitoring: http://localhost:3001"
