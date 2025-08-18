#!/bin/bash

# Test Runner Script for AGI Evaluation Sandbox
# Provides different test execution modes and reporting options

set -euo pipefail

# Colors
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

# Default values
TEST_TYPE="all"
COVERAGE=true
PARALLEL=true
VERBOSE=false
BAIL=false
OUTPUT_FORMAT="terminal"
REPORT_DIR="test-reports"
MARKERS=""
EXCLUDE_MARKERS=""

# Help function
show_help() {
    cat << EOF
Test Runner for AGI Evaluation Sandbox

Usage: $0 [OPTIONS]

OPTIONS:
    -t, --type TYPE         Test type: unit, integration, e2e, smoke, all (default: all)
    -m, --markers MARKERS   Run tests with specific markers (comma-separated)
    -x, --exclude MARKERS   Exclude tests with specific markers (comma-separated)
    -c, --coverage          Enable coverage reporting (default: true)
    --no-coverage           Disable coverage reporting
    -p, --parallel          Run tests in parallel (default: true)
    --no-parallel           Run tests sequentially
    -v, --verbose           Verbose output
    -b, --bail             Stop on first failure
    -o, --output FORMAT     Output format: terminal, junit, html (default: terminal)
    -r, --report-dir DIR    Report directory (default: test-reports)
    -h, --help             Show this help message

EXAMPLES:
    $0                                    # Run all tests
    $0 -t unit                           # Run only unit tests
    $0 -t integration -v                 # Run integration tests with verbose output
    $0 -m "smoke,api" --no-coverage     # Run smoke and API tests without coverage
    $0 -x "slow,external" -p            # Run all tests except slow and external ones
    $0 -o junit -r reports              # Generate JUnit reports in reports directory
    
TEST MARKERS:
    unit            - Unit tests (fast, isolated)
    integration     - Integration tests (slower, external dependencies)
    e2e            - End-to-end tests (slowest, full system)
    smoke          - Smoke tests (critical functionality)
    slow           - Slow tests (> 1 second)
    api            - API endpoint tests
    database       - Database interaction tests
    external       - Tests requiring external services
    benchmark      - Benchmark evaluation tests
    security       - Security-related tests
    performance    - Performance tests
    regression     - Regression tests
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -t|--type)
                TEST_TYPE="$2"
                shift 2
                ;;
            -m|--markers)
                MARKERS="$2"
                shift 2
                ;;
            -x|--exclude)
                EXCLUDE_MARKERS="$2"
                shift 2
                ;;
            -c|--coverage)
                COVERAGE=true
                shift
                ;;
            --no-coverage)
                COVERAGE=false
                shift
                ;;
            -p|--parallel)
                PARALLEL=true
                shift
                ;;
            --no-parallel)
                PARALLEL=false
                shift
                ;;
            -v|--verbose)
                VERBOSE=true
                shift
                ;;
            -b|--bail)
                BAIL=true
                shift
                ;;
            -o|--output)
                OUTPUT_FORMAT="$2"
                shift 2
                ;;
            -r|--report-dir)
                REPORT_DIR="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate arguments
validate_args() {
    # Validate test type
    case $TEST_TYPE in
        unit|integration|e2e|smoke|all) ;;
        *)
            log_error "Invalid test type: $TEST_TYPE"
            log_error "Valid types: unit, integration, e2e, smoke, all"
            exit 1
            ;;
    esac
    
    # Validate output format
    case $OUTPUT_FORMAT in
        terminal|junit|html) ;;
        *)
            log_error "Invalid output format: $OUTPUT_FORMAT"
            log_error "Valid formats: terminal, junit, html"
            exit 1
            ;;
    esac
}

# Build pytest command
build_pytest_command() {
    local cmd="python -m pytest"
    
    # Test paths based on type
    case $TEST_TYPE in
        unit)
            cmd="$cmd tests/unit/"
            ;;
        integration)
            cmd="$cmd tests/integration/"
            ;;
        e2e)
            cmd="$cmd tests/e2e/"
            ;;
        smoke)
            cmd="$cmd -m smoke"
            ;;
        all)
            cmd="$cmd tests/"
            ;;
    esac
    
    # Add markers
    if [[ -n "$MARKERS" ]]; then
        cmd="$cmd -m \"$MARKERS\""
    fi
    
    # Exclude markers
    if [[ -n "$EXCLUDE_MARKERS" ]]; then
        IFS=',' read -ra EXCLUDE_ARRAY <<< "$EXCLUDE_MARKERS"
        for marker in "${EXCLUDE_ARRAY[@]}"; do
            cmd="$cmd -m \"not $marker\""
        done
    fi
    
    # Coverage options
    if [[ "$COVERAGE" == true ]]; then
        cmd="$cmd --cov=src --cov-report=html:$REPORT_DIR/htmlcov --cov-report=xml:$REPORT_DIR/coverage.xml --cov-report=term-missing"
    else
        cmd="$cmd --no-cov"
    fi
    
    # Parallel execution
    if [[ "$PARALLEL" == true ]]; then
        cmd="$cmd -n auto"
    fi
    
    # Verbose output
    if [[ "$VERBOSE" == true ]]; then
        cmd="$cmd -v"
    fi
    
    # Stop on first failure
    if [[ "$BAIL" == true ]]; then
        cmd="$cmd -x"
    fi
    
    # Output format
    case $OUTPUT_FORMAT in
        junit)
            cmd="$cmd --junit-xml=$REPORT_DIR/junit.xml"
            ;;
        html)
            cmd="$cmd --html=$REPORT_DIR/report.html --self-contained-html"
            ;;
    esac
    
    echo "$cmd"
}

# Setup environment
setup_environment() {
    log_info "Setting up test environment..."
    
    # Create report directory
    mkdir -p "$REPORT_DIR"
    
    # Activate virtual environment if it exists
    if [[ -f "venv/bin/activate" ]]; then
        log_info "Activating virtual environment..."
        source venv/bin/activate
    fi
    
    # Install test dependencies if needed
    if [[ ! -f ".test-deps-installed" ]]; then
        log_info "Installing test dependencies..."
        pip install -e ".[dev]" >/dev/null 2>&1 || log_warning "Failed to install dependencies"
        touch ".test-deps-installed"
    fi
    
    # Start test services if needed
    if [[ "$TEST_TYPE" == "integration" || "$TEST_TYPE" == "e2e" || "$TEST_TYPE" == "all" ]]; then
        start_test_services
    fi
}

# Start test services
start_test_services() {
    log_info "Starting test services..."
    
    # Check if Docker is available
    if command -v docker-compose >/dev/null 2>&1; then
        if [[ -f "docker-compose.test.yml" ]]; then
            log_info "Starting Docker test services..."
            docker-compose -f docker-compose.test.yml up -d >/dev/null 2>&1 || log_warning "Failed to start Docker services"
            sleep 5  # Wait for services to be ready
        fi
    fi
}

# Stop test services
stop_test_services() {
    log_info "Stopping test services..."
    
    if command -v docker-compose >/dev/null 2>&1; then
        if [[ -f "docker-compose.test.yml" ]]; then
            docker-compose -f docker-compose.test.yml down >/dev/null 2>&1 || log_warning "Failed to stop Docker services"
        fi
    fi
}

# Run tests
run_tests() {
    local pytest_cmd
    pytest_cmd=$(build_pytest_command)
    
    log_info "Running $TEST_TYPE tests..."
    log_info "Command: $pytest_cmd"
    
    # Run tests
    eval "$pytest_cmd"
    local exit_code=$?
    
    return $exit_code
}

# Generate summary
generate_summary() {
    local exit_code=$1
    
    echo ""
    log_info "Test Summary"
    echo "============"
    
    if [[ $exit_code -eq 0 ]]; then
        log_success "All tests passed!"
    else
        log_error "Some tests failed (exit code: $exit_code)"
    fi
    
    # Show coverage summary if enabled
    if [[ "$COVERAGE" == true && -f "$REPORT_DIR/coverage.xml" ]]; then
        log_info "Coverage report generated: $REPORT_DIR/htmlcov/index.html"
    fi
    
    # Show other reports
    case $OUTPUT_FORMAT in
        junit)
            log_info "JUnit report generated: $REPORT_DIR/junit.xml"
            ;;
        html)
            log_info "HTML report generated: $REPORT_DIR/report.html"
            ;;
    esac
    
    echo ""
}

# Cleanup
cleanup() {
    if [[ "$TEST_TYPE" == "integration" || "$TEST_TYPE" == "e2e" || "$TEST_TYPE" == "all" ]]; then
        stop_test_services
    fi
}

# Main execution
main() {
    parse_args "$@"
    validate_args
    
    log_info "Starting test execution..."
    log_info "Test type: $TEST_TYPE"
    log_info "Coverage: $COVERAGE"
    log_info "Parallel: $PARALLEL"
    log_info "Output format: $OUTPUT_FORMAT"
    
    # Trap cleanup on exit
    trap cleanup EXIT
    
    setup_environment
    
    run_tests
    local exit_code=$?
    
    generate_summary $exit_code
    
    exit $exit_code
}

# Run main function
main "$@"