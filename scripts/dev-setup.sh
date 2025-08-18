#!/bin/bash

# Development Environment Setup Script
# AGI Evaluation Sandbox

set -euo pipefail

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

# Check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Main setup function
main() {
    log_info "Starting development environment setup for AGI Evaluation Sandbox"
    
    # Check system requirements
    check_system_requirements
    
    # Setup Python environment
    setup_python_env
    
    # Setup Node.js environment
    setup_node_env
    
    # Setup pre-commit hooks
    setup_pre_commit
    
    # Setup database
    setup_database
    
    # Verify installation
    verify_installation
    
    log_success "Development environment setup complete!"
    print_next_steps
}

check_system_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python
    if ! command_exists python3; then
        log_error "Python 3 is required but not installed"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2 | cut -d'.' -f1-2)
    if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
        log_error "Python 3.8+ is required, found $PYTHON_VERSION"
        exit 1
    fi
    log_success "Python $PYTHON_VERSION found"
    
    # Check Node.js
    if ! command_exists node; then
        log_error "Node.js is required but not installed"
        exit 1
    fi
    
    NODE_VERSION=$(node --version)
    if ! node -e "process.exit(parseInt(process.version.slice(1)) >= 18 ? 0 : 1)"; then
        log_error "Node.js 18+ is required, found $NODE_VERSION"
        exit 1
    fi
    log_success "Node.js $NODE_VERSION found"
    
    # Check Docker
    if command_exists docker; then
        log_success "Docker found"
    else
        log_warning "Docker not found - some features will be unavailable"
    fi
    
    # Check Git
    if ! command_exists git; then
        log_error "Git is required but not installed"
        exit 1
    fi
    log_success "Git found"
}

setup_python_env() {
    log_info "Setting up Python environment..."
    
    # Create virtual environment if it doesn't exist
    if [[ ! -d "venv" ]]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip
    
    # Install development dependencies
    log_info "Installing Python dependencies..."
    pip install -e ".[dev]"
    
    log_success "Python environment setup complete"
}

setup_node_env() {
    log_info "Setting up Node.js environment..."
    
    # Install root dependencies
    if [[ -f "package.json" ]]; then
        log_info "Installing root npm dependencies..."
        npm install
    fi
    
    # Install dashboard dependencies
    if [[ -d "dashboard" && -f "dashboard/package.json" ]]; then
        log_info "Installing dashboard dependencies..."
        cd dashboard
        npm install
        cd ..
    fi
    
    log_success "Node.js environment setup complete"
}

setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Install pre-commit hooks
    if [[ -f ".pre-commit-config.yaml" ]]; then
        pre-commit install
        pre-commit install --hook-type commit-msg
        log_success "Pre-commit hooks installed"
    else
        log_warning "No .pre-commit-config.yaml found, skipping pre-commit setup"
    fi
}

setup_database() {
    log_info "Setting up database..."
    
    # Check if Docker is available for database setup
    if command_exists docker && command_exists docker-compose; then
        if [[ -f "docker-compose.yml" ]]; then
            log_info "Starting database services with Docker Compose..."
            docker-compose up -d db redis
            
            # Wait for database to be ready
            log_info "Waiting for database to be ready..."
            sleep 10
            
            # Run migrations
            log_info "Running database migrations..."
            source venv/bin/activate
            npm run db:migrate 2>/dev/null || log_warning "Database migration failed - you may need to run it manually"
            
            log_success "Database setup complete"
        else
            log_warning "No docker-compose.yml found, skipping database setup"
        fi
    else
        log_warning "Docker not available, skipping database setup"
    fi
}

verify_installation() {
    log_info "Verifying installation..."
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Test Python imports
    log_info "Testing Python imports..."
    python3 -c "import agi_eval_sandbox; print('✓ AGI Eval Sandbox imported successfully')" || log_warning "Python import test failed"
    
    # Test CLI
    log_info "Testing CLI..."
    python3 -m agi_eval_sandbox.cli --help >/dev/null && log_success "CLI test passed" || log_warning "CLI test failed"
    
    # Test lint commands
    log_info "Testing linting tools..."
    npm run lint:api >/dev/null 2>&1 && log_success "Python linting works" || log_warning "Python linting failed"
    
    if [[ -d "dashboard" ]]; then
        cd dashboard
        npm run lint >/dev/null 2>&1 && log_success "Frontend linting works" || log_warning "Frontend linting failed"
        cd ..
    fi
}

print_next_steps() {
    echo ""
    log_info "Next steps:"
    echo "1. Activate the virtual environment: source venv/bin/activate"
    echo "2. Start the development server: npm run dev"
    echo "3. Access the dashboard at: http://localhost:8080"
    echo "4. Access the API docs at: http://localhost:8000/docs"
    echo "5. Run tests: npm test"
    echo "6. Check the README.md for more information"
    echo ""
    log_info "Environment variables:"
    if [[ -f ".env.example" ]]; then
        echo "Copy .env.example to .env and configure your settings:"
        echo "cp .env.example .env"
    fi
    echo ""
    log_info "Development workflow:"
    echo "- Use 'npm run dev' to start both API and dashboard in development mode"
    echo "- Use 'npm run lint' to check code quality"
    echo "- Use 'npm run test' to run all tests"
    echo "- Use 'git commit' to commit changes (pre-commit hooks will run)"
    echo ""
}

# Run main function
main "$@"