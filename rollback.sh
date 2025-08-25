#!/bin/bash
set -e

echo "🔄 Starting rollback procedure..."

ENVIRONMENT=${1:-production}
PREVIOUS_VERSION=${2:-previous}

echo "Environment: $ENVIRONMENT"
echo "Rolling back to: $PREVIOUS_VERSION"

# Stop current deployment
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f docker-compose.prod.yml down
elif [ "$ENVIRONMENT" = "staging" ]; then
    docker-compose -f docker-compose.staging.yml down
else
    docker-compose down
fi

# Deploy previous version
echo "🏗️  Deploying previous version..."
if [ "$ENVIRONMENT" = "production" ]; then
    docker-compose -f docker-compose.prod.yml up -d
elif [ "$ENVIRONMENT" = "staging" ]; then
    docker-compose -f docker-compose.staging.yml up -d
else
    docker-compose up -d
fi

echo "✅ Rollback completed"
