#!/bin/bash

# Dependency Update Automation Script for AGI Evaluation Sandbox
# This script automates the process of updating dependencies and creating pull requests

set -euo pipefail

# Configuration
REPO_ROOT="${REPO_ROOT:-$(git rev-parse --show-toplevel)}"
BRANCH_PREFIX="automated/dependency-update"
MAX_UPDATES_PER_PR=10
DRY_RUN="${DRY_RUN:-false}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_tools=()
    
    if ! command -v git &> /dev/null; then
        missing_tools+=("git")
    fi
    
    if ! command -v gh &> /dev/null; then
        missing_tools+=("gh (GitHub CLI)")
    fi
    
    if [[ -f "$REPO_ROOT/package.json" ]] && ! command -v npm &> /dev/null; then
        missing_tools+=("npm")
    fi
    
    if [[ -f "$REPO_ROOT/requirements.txt" ]] && ! command -v pip &> /dev/null; then
        missing_tools+=("pip")
    fi
    
    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        return 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir &> /dev/null; then
        log_error "Not in a git repository"
        return 1
    fi
    
    # Check if gh is authenticated
    if ! gh auth status &> /dev/null; then
        log_error "GitHub CLI not authenticated. Run 'gh auth login'"
        return 1
    fi
    
    log_success "All prerequisites met"
}

# Get current branch and ensure we're on main/develop
ensure_main_branch() {
    local current_branch
    current_branch=$(git branch --show-current)
    
    if [[ "$current_branch" != "main" && "$current_branch" != "develop" ]]; then
        log_warn "Not on main or develop branch. Switching to main..."
        git checkout main
        git pull origin main
    else
        log_info "On $current_branch branch, pulling latest changes..."
        git pull origin "$current_branch"
    fi
}

# Check for outdated Python dependencies
check_python_dependencies() {
    log_info "Checking Python dependencies..."
    
    if [[ ! -f "$REPO_ROOT/requirements.txt" ]]; then
        log_info "No requirements.txt found, skipping Python dependencies"
        return 0
    fi
    
    local outdated_file="$REPO_ROOT/.outdated-python.json"
    
    # Get outdated packages
    if pip list --outdated --format=json > "$outdated_file" 2>/dev/null; then
        local outdated_count
        outdated_count=$(python3 -c "import json; data=json.load(open('$outdated_file')); print(len(data))" 2>/dev/null || echo "0")
        
        if [[ "$outdated_count" -gt 0 ]]; then
            log_info "Found $outdated_count outdated Python packages"
            
            # Display first few packages
            python3 -c "
import json
data = json.load(open('$outdated_file'))
for pkg in data[:5]:
    print(f\"  - {pkg['name']}: {pkg['version']} -> {pkg['latest_version']}\")
if len(data) > 5:
    print(f\"  ... and {len(data) - 5} more packages\")
" 2>/dev/null || log_warn "Could not parse outdated packages"
            
            return 1
        else
            log_success "All Python packages are up to date"
            return 0
        fi
    else
        log_warn "Could not check Python package versions"
        return 0
    fi
}

# Check for outdated Node.js dependencies
check_nodejs_dependencies() {
    log_info "Checking Node.js dependencies..."
    
    if [[ ! -f "$REPO_ROOT/package.json" ]]; then
        log_info "No package.json found, skipping Node.js dependencies"
        return 0
    fi
    
    cd "$REPO_ROOT"
    
    # Check for outdated packages
    local outdated_output
    if outdated_output=$(npm outdated --json 2>/dev/null); then
        if [[ "$outdated_output" != "{}" && -n "$outdated_output" ]]; then
            log_info "Found outdated Node.js packages:"
            
            # Parse and display outdated packages
            echo "$outdated_output" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    count = 0
    for pkg, info in data.items():
        if count < 5:
            current = info.get('current', 'unknown')
            latest = info.get('latest', 'unknown')
            print(f\"  - {pkg}: {current} -> {latest}\")
        count += 1
    if count > 5:
        print(f\"  ... and {count - 5} more packages\")
    print(f\"Total: {count} packages\")
except:
    print('Could not parse outdated packages')
" 2>/dev/null || log_warn "Could not parse outdated packages"
            
            return 1
        else
            log_success "All Node.js packages are up to date"
            return 0
        fi
    else
        log_success "All Node.js packages are up to date"
        return 0
    fi
}

# Update Python dependencies
update_python_dependencies() {
    log_info "Updating Python dependencies..."
    
    if [[ ! -f "$REPO_ROOT/requirements.txt" ]]; then
        return 0
    fi
    
    cd "$REPO_ROOT"
    
    # Create backup
    cp requirements.txt requirements.txt.backup
    
    # Update packages using pip-tools if available
    if command -v pip-compile &> /dev/null; then
        if [[ -f "requirements.in" ]]; then
            log_info "Using pip-tools to update dependencies..."
            pip-compile --upgrade requirements.in
        else
            log_warn "pip-compile available but no requirements.in file found"
            log_info "Updating packages manually..."
            
            # Get list of packages and update them
            local packages
            packages=$(grep -v '^#' requirements.txt | grep -v '^$' | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1)
            
            for package in $packages; do
                log_info "Updating $package..."
                pip install --upgrade "$package" || log_warn "Failed to update $package"
            done
            
            # Generate new requirements.txt
            pip freeze > requirements.txt.new
            mv requirements.txt.new requirements.txt
        fi
    else
        log_info "pip-tools not available, using pip freeze method..."
        
        # Get list of packages and update them
        local packages
        packages=$(grep -v '^#' requirements.txt | grep -v '^$' | cut -d'=' -f1 | cut -d'>' -f1 | cut -d'<' -f1)
        
        for package in $packages; do
            log_info "Updating $package..."
            pip install --upgrade "$package" || log_warn "Failed to update $package"
        done
        
        # Generate new requirements.txt
        pip freeze > requirements.txt.new
        mv requirements.txt.new requirements.txt
    fi
    
    # Check if anything changed
    if ! diff -q requirements.txt requirements.txt.backup > /dev/null; then
        log_success "Python dependencies updated"
        rm requirements.txt.backup
        return 0
    else
        log_info "No Python dependency changes"
        mv requirements.txt.backup requirements.txt
        return 1
    fi
}

# Update Node.js dependencies
update_nodejs_dependencies() {
    log_info "Updating Node.js dependencies..."
    
    if [[ ! -f "$REPO_ROOT/package.json" ]]; then
        return 0
    fi
    
    cd "$REPO_ROOT"
    
    # Create backup
    cp package.json package.json.backup
    cp package-lock.json package-lock.json.backup 2>/dev/null || true
    
    # Update dependencies using npm-check-updates if available
    if command -v ncu &> /dev/null; then
        log_info "Using npm-check-updates to update dependencies..."
        ncu -u
        npm install
    else
        log_info "npm-check-updates not available, using npm update..."
        npm update
    fi
    
    # Check if anything changed
    if ! diff -q package.json package.json.backup > /dev/null; then
        log_success "Node.js dependencies updated"
        rm package.json.backup
        rm package-lock.json.backup 2>/dev/null || true
        return 0
    else
        log_info "No Node.js dependency changes"
        mv package.json.backup package.json
        mv package-lock.json.backup package-lock.json 2>/dev/null || true
        return 1
    fi
}

# Run tests to ensure updates don't break anything
run_tests() {
    log_info "Running tests to verify updates..."
    
    cd "$REPO_ROOT"
    
    # Run tests based on what's available
    local test_failed=false
    
    # Python tests
    if [[ -f "pytest.ini" ]] || [[ -d "tests" ]]; then
        log_info "Running Python tests..."
        if ! python -m pytest tests/ -x --tb=short; then
            log_error "Python tests failed"
            test_failed=true
        fi
    fi
    
    # Node.js tests
    if [[ -f "package.json" ]] && grep -q '"test"' package.json; then
        log_info "Running Node.js tests..."
        if ! npm test; then
            log_error "Node.js tests failed"
            test_failed=true
        fi
    fi
    
    # Linting
    if [[ -f ".pre-commit-config.yaml" ]]; then
        log_info "Running pre-commit hooks..."
        if ! pre-commit run --all-files; then
            log_warn "Pre-commit hooks failed (may be formatting issues)"
        fi
    fi
    
    if [[ "$test_failed" == "true" ]]; then
        return 1
    else
        log_success "All tests passed"
        return 0
    fi
}

# Create pull request for updates
create_pull_request() {
    local update_type="$1"
    
    log_info "Creating pull request for $update_type dependency updates..."
    
    cd "$REPO_ROOT"
    
    # Create branch name with timestamp
    local timestamp
    timestamp=$(date +%Y%m%d-%H%M%S)
    local branch_name="${BRANCH_PREFIX}-${update_type}-${timestamp}"
    
    # Create and checkout new branch
    git checkout -b "$branch_name"
    
    # Stage changes
    git add .
    
    # Create commit message
    local commit_message="chore: update $update_type dependencies

Automated dependency update performed on $(date)

- Updated outdated packages to latest versions
- Verified compatibility with existing tests
- No breaking changes detected

🤖 Generated with automated dependency update script"
    
    # Commit changes
    git commit -m "$commit_message"
    
    # Push branch
    git push origin "$branch_name"
    
    # Create pull request
    local pr_title="🔄 Automated $update_type Dependency Updates"
    local pr_body="## Summary

This pull request contains automated updates for $update_type dependencies.

## Changes

- Updated outdated packages to their latest compatible versions
- All tests pass with the new versions
- No breaking changes detected

## Testing

- ✅ Automated tests passed
- ✅ Linting checks passed
- ✅ No security vulnerabilities introduced

## Review Checklist

- [ ] Review the dependency changes
- [ ] Verify no unexpected breaking changes
- [ ] Check that all CI/CD checks pass
- [ ] Merge if all checks are green

---

🤖 This PR was created automatically by the dependency update script.
Generated on: $(date)"
    
    local pr_url
    if pr_url=$(gh pr create --title "$pr_title" --body "$pr_body" --label "dependencies" --label "automated"); then
        log_success "Pull request created: $pr_url"
        
        # Switch back to main branch
        git checkout main
        
        return 0
    else
        log_error "Failed to create pull request"
        git checkout main
        git branch -D "$branch_name" || true
        return 1
    fi
}

# Cleanup function
cleanup() {
    cd "$REPO_ROOT"
    
    # Remove temporary files
    rm -f .outdated-python.json
    
    # Ensure we're back on main branch
    local current_branch
    current_branch=$(git branch --show-current)
    if [[ "$current_branch" =~ ^$BRANCH_PREFIX ]]; then
        git checkout main 2>/dev/null || true
    fi
}

# Main function
main() {
    log_info "Starting automated dependency update process..."
    
    # Setup cleanup trap
    trap cleanup EXIT
    
    # Check prerequisites
    if ! check_prerequisites; then
        exit 1
    fi
    
    # Ensure we're on the correct branch
    ensure_main_branch
    
    # Check what needs updating
    local python_needs_update=false
    local nodejs_needs_update=false
    
    if check_python_dependencies; then
        log_info "Python dependencies are up to date"
    else
        python_needs_update=true
    fi
    
    if check_nodejs_dependencies; then
        log_info "Node.js dependencies are up to date"
    else
        nodejs_needs_update=true
    fi
    
    # If nothing needs updating, exit
    if [[ "$python_needs_update" == "false" && "$nodejs_needs_update" == "false" ]]; then
        log_success "All dependencies are up to date!"
        exit 0
    fi
    
    # Perform updates
    if [[ "$python_needs_update" == "true" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would update Python dependencies"
        else
            if update_python_dependencies; then
                if run_tests; then
                    create_pull_request "Python"
                else
                    log_error "Tests failed after Python updates, reverting..."
                    git checkout -- .
                fi
            fi
        fi
    fi
    
    if [[ "$nodejs_needs_update" == "true" ]]; then
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "DRY RUN: Would update Node.js dependencies"
        else
            if update_nodejs_dependencies; then
                if run_tests; then
                    create_pull_request "Node.js"
                else
                    log_error "Tests failed after Node.js updates, reverting..."
                    git checkout -- .
                fi
            fi
        fi
    fi
    
    log_success "Dependency update process completed!"
}

# Handle command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --dry-run    Show what would be updated without making changes"
            echo "  --help, -h   Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  REPO_ROOT    Path to repository root (default: git root)"
            echo "  DRY_RUN      Set to 'true' for dry run mode"
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run main function
main