#!/bin/bash
set -e

# AGI Evaluation Sandbox - Docker Entrypoint
# ===========================================

# Colors for output
BLUE='\033[36m'
GREEN='\033[32m'
YELLOW='\033[33m'
RED='\033[31m'
RESET='\033[0m'

# Logging function
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')] $1${RESET}"
}

log_success() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${RESET}"
}

log_warning() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] $1${RESET}"
}

log_error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] $1${RESET}"
}

# Environment validation
validate_environment() {
    log "Validating environment..."
    
    # Check required environment variables
    local required_vars=(
        "DATABASE_URL"
        "REDIS_URL"
        "SECRET_KEY"
    )
    
    for var in "${required_vars[@]}"; do
        if [ -z "${!var}" ]; then
            log_error "Required environment variable $var is not set"
            exit 1
        fi
    done
    
    log_success "Environment validation complete"
}

# Wait for services
wait_for_service() {
    local service=$1
    local host=$2
    local port=$3
    local timeout=${4:-30}
    
    log "Waiting for $service at $host:$port (timeout: ${timeout}s)..."
    
    local counter=0
    while ! nc -z "$host" "$port" 2>/dev/null; do
        if [ $counter -ge $timeout ]; then
            log_error "Timeout waiting for $service"
            exit 1
        fi
        sleep 1
        counter=$((counter + 1))
    done
    
    log_success "$service is ready"
}

# Database setup
setup_database() {
    if [ "$SKIP_DB_INIT" = "true" ]; then
        log_warning "Skipping database initialization"
        return
    fi
    
    log "Setting up database..."
    
    # Extract database connection details
    if [[ "$DATABASE_URL" =~ postgresql://([^:]+):([^@]+)@([^:]+):([0-9]+)/(.+) ]]; then
        local db_host="${BASH_REMATCH[3]}"
        local db_port="${BASH_REMATCH[4]}"
        
        wait_for_service "PostgreSQL" "$db_host" "$db_port"
        
        # Run database migrations
        log "Running database migrations..."
        python -m alembic upgrade head || {
            log_error "Database migration failed"
            exit 1
        }
        
        log_success "Database setup complete"
    else
        log_error "Invalid DATABASE_URL format"
        exit 1
    fi
}

# Redis setup
setup_redis() {
    log "Setting up Redis..."
    
    # Extract Redis connection details
    if [[ "$REDIS_URL" =~ redis://(:([^@]+)@)?([^:]+):([0-9]+)/([0-9]+) ]]; then
        local redis_host="${BASH_REMATCH[3]}"
        local redis_port="${BASH_REMATCH[4]}"
        
        wait_for_service "Redis" "$redis_host" "$redis_port"
        log_success "Redis setup complete"
    else
        log_error "Invalid REDIS_URL format"
        exit 1
    fi
}

# Start application server
start_server() {
    log "Starting AGI Evaluation Sandbox API server..."
    exec uvicorn main:app \
        --host 0.0.0.0 \
        --port "${PORT:-8000}" \
        --workers "${WORKERS:-1}" \
        --log-level "${LOG_LEVEL:-info}" \
        --access-log \
        --use-colors
}

# Start Celery worker
start_worker() {
    log "Starting Celery worker..."
    
    wait_for_service "Redis" "$(echo $CELERY_BROKER_URL | sed -n 's/.*@\([^:]*\):\([0-9]*\).*/\1/p')" "$(echo $CELERY_BROKER_URL | sed -n 's/.*@\([^:]*\):\([0-9]*\).*/\2/p')"
    
    exec celery -A src.worker worker \
        --loglevel="${LOG_LEVEL:-info}" \
        --concurrency="${WORKER_CONCURRENCY:-2}" \
        --max-tasks-per-child=1000 \
        --time-limit=3600 \
        --soft-time-limit=3300
}

# Start Celery beat scheduler
start_scheduler() {
    log "Starting Celery beat scheduler..."
    
    wait_for_service "Redis" "$(echo $CELERY_BROKER_URL | sed -n 's/.*@\([^:]*\):\([0-9]*\).*/\1/p')" "$(echo $CELERY_BROKER_URL | sed -n 's/.*@\([^:]*\):\([0-9]*\).*/\2/p')"
    
    exec celery -A src.worker beat \
        --loglevel="${LOG_LEVEL:-info}" \
        --schedule=/tmp/celerybeat-schedule \
        --pidfile=/tmp/celerybeat.pid
}

# Start Flower monitoring
start_flower() {
    log "Starting Flower monitoring..."
    
    exec celery -A src.worker flower \
        --port="${FLOWER_PORT:-5555}" \
        --basic_auth="${FLOWER_USER:-admin}:${FLOWER_PASSWORD:-admin}"
}

# Run migrations only
run_migrations() {
    validate_environment
    setup_database
    log_success "Migrations completed successfully"
}

# Main entry point
main() {
    local command=${1:-serve}
    
    log "AGI Evaluation Sandbox starting..."
    log "Command: $command"
    log "Environment: ${ENVIRONMENT:-development}"
    
    case "$command" in
        serve)
            validate_environment
            setup_database
            setup_redis
            start_server
            ;;
        worker)
            validate_environment
            setup_redis
            start_worker
            ;;
        scheduler)
            validate_environment
            setup_redis
            start_scheduler
            ;;
        flower)
            validate_environment
            setup_redis
            start_flower
            ;;
        migrate)
            run_migrations
            ;;
        shell)
            validate_environment
            log "Starting interactive shell..."
            exec python -i -c "
import asyncio
from src.database import get_db
from src.models import *
print('AGI Evaluation Sandbox shell ready!')
print('Available: get_db, models, asyncio')
"
            ;;
        test)
            log "Running tests..."
            exec python -m pytest tests/ -v
            ;;
        *)
            log "Running custom command: $*"
            exec "$@"
            ;;
    esac
}

# Handle signals gracefully
trap 'log "Received SIGTERM, shutting down gracefully..."; exit 0' SIGTERM
trap 'log "Received SIGINT, shutting down gracefully..."; exit 0' SIGINT

# Run main function
main "$@"