#!/bin/bash

# Release Automation Script
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
DOCKER_REGISTRY="ghcr.io"
DOCKER_IMAGE="agi-eval-sandbox"
VERSION_TYPE="patch"
BRANCH="main"
SKIP_TESTS=false
SKIP_BUILD=false
SKIP_PUSH=false
DRY_RUN=false

# Help function
show_help() {
    cat << EOF
Release Automation Script for AGI Evaluation Sandbox

Usage: $0 [OPTIONS]

OPTIONS:
    -v, --version TYPE      Version bump type: major, minor, patch (default: patch)
    -b, --branch BRANCH     Release branch (default: main)
    -r, --registry URL      Docker registry (default: ghcr.io)
    --skip-tests           Skip running tests
    --skip-build           Skip building Docker images
    --skip-push            Skip pushing to registry
    --dry-run              Show what would be done without executing
    -h, --help             Show this help message

EXAMPLES:
    $0                                    # Create patch release from main
    $0 -v minor                          # Create minor version bump
    $0 -v major --dry-run               # Preview major release
    $0 --skip-tests --skip-build        # Create tag only release
EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -v|--version)
                VERSION_TYPE="$2"
                shift 2
                ;;
            -b|--branch)
                BRANCH="$2"
                shift 2
                ;;
            -r|--registry)
                DOCKER_REGISTRY="$2"
                shift 2
                ;;
            --skip-tests)
                SKIP_TESTS=true
                shift
                ;;
            --skip-build)
                SKIP_BUILD=true
                shift
                ;;
            --skip-push)
                SKIP_PUSH=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
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
    case $VERSION_TYPE in
        major|minor|patch) ;;
        *)
            log_error "Invalid version type: $VERSION_TYPE"
            log_error "Valid types: major, minor, patch"
            exit 1
            ;;
    esac
}

# Execute command (respecting dry-run)
execute() {
    local cmd="$1"
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would execute: $cmd"
    else
        eval "$cmd"
    fi
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check git
    if ! command -v git >/dev/null 2>&1; then
        log_error "Git is required but not installed"
        exit 1
    fi
    
    # Check if we're in a git repository
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
        log_error "Not in a git repository"
        exit 1
    fi
    
    # Check working directory is clean
    if [[ -n $(git status --porcelain) ]]; then
        log_error "Working directory is not clean. Please commit or stash changes."
        exit 1
    fi
    
    # Check we're on the correct branch
    current_branch=$(git branch --show-current)
    if [[ "$current_branch" != "$BRANCH" ]]; then
        log_error "Not on release branch '$BRANCH'. Currently on '$current_branch'"
        exit 1
    fi
    
    # Check if branch is up to date with remote
    git fetch origin "$BRANCH"
    if [[ $(git rev-parse HEAD) != $(git rev-parse "origin/$BRANCH") ]]; then
        log_error "Branch is not up to date with remote. Please pull latest changes."
        exit 1
    fi
    
    # Check Docker if needed
    if [[ "$SKIP_BUILD" == "false" ]] && ! command -v docker >/dev/null 2>&1; then
        log_error "Docker is required for building but not installed"
        exit 1
    fi
    
    # Check semantic-release if available
    if command -v semantic-release >/dev/null 2>&1; then
        log_info "semantic-release available for automated versioning"
    fi
    
    log_success "Prerequisites check passed"
}

# Get current version
get_current_version() {
    # Try to get version from git tags
    if git describe --tags --exact-match HEAD >/dev/null 2>&1; then
        git describe --tags --exact-match HEAD
    elif git describe --tags >/dev/null 2>&1; then
        git describe --tags
    else
        echo "0.0.0"
    fi
}

# Calculate next version
calculate_next_version() {
    local current_version="$1"
    
    # Remove 'v' prefix if present
    current_version=${current_version#v}
    
    # Split version into parts
    IFS='.' read -ra VERSION_PARTS <<< "$current_version"
    major=${VERSION_PARTS[0]:-0}
    minor=${VERSION_PARTS[1]:-0}
    patch=${VERSION_PARTS[2]:-0}
    
    case $VERSION_TYPE in
        major)
            major=$((major + 1))
            minor=0
            patch=0
            ;;
        minor)
            minor=$((minor + 1))
            patch=0
            ;;
        patch)
            patch=$((patch + 1))
            ;;
    esac
    
    echo "v$major.$minor.$patch"
}

# Run tests
run_tests() {
    if [[ "$SKIP_TESTS" == "true" ]]; then
        log_warning "Skipping tests as requested"
        return
    fi
    
    log_info "Running test suite..."
    
    # Run linting
    log_info "Running linting..."
    execute "npm run lint"
    
    # Run type checking
    log_info "Running type checking..."
    execute "npm run typecheck"
    
    # Run unit tests
    log_info "Running unit tests..."
    execute "npm run test:api"
    execute "npm run test:dashboard"
    
    # Run integration tests
    log_info "Running integration tests..."
    execute "npm run test:e2e"
    
    log_success "All tests passed"
}

# Build artifacts
build_artifacts() {
    if [[ "$SKIP_BUILD" == "true" ]]; then
        log_warning "Skipping build as requested"
        return
    fi
    
    log_info "Building artifacts..."
    
    # Build Python package
    log_info "Building Python package..."
    execute "cd api && python -m build"
    
    # Build frontend
    log_info "Building frontend..."
    execute "npm run build:dashboard"
    
    # Build Docker images
    local next_version="$1"
    log_info "Building Docker image with tag: $next_version"
    execute "docker build -t $DOCKER_REGISTRY/$DOCKER_IMAGE:$next_version ."
    execute "docker tag $DOCKER_REGISTRY/$DOCKER_IMAGE:$next_version $DOCKER_REGISTRY/$DOCKER_IMAGE:latest"
    
    log_success "Artifacts built successfully"
}

# Generate SBOM
generate_sbom() {
    log_info "Generating Software Bill of Materials (SBOM)..."
    
    if [[ -x "./scripts/generate-sbom.sh" ]]; then
        execute "./scripts/generate-sbom.sh"
    else
        log_warning "SBOM generation script not found or not executable"
    fi
}

# Update version in files
update_version_files() {
    local next_version="$1"
    local version_without_v="${next_version#v}"
    
    log_info "Updating version in project files..."
    
    # Update package.json
    if [[ -f "package.json" ]]; then
        log_info "Updating package.json version to $version_without_v"
        if [[ "$DRY_RUN" == "false" ]]; then
            sed -i "s/\"version\": \".*\"/\"version\": \"$version_without_v\"/" package.json
        fi
    fi
    
    # Update pyproject.toml
    if [[ -f "pyproject.toml" ]]; then
        log_info "Updating pyproject.toml version to $version_without_v"
        if [[ "$DRY_RUN" == "false" ]]; then
            sed -i "s/version = \".*\"/version = \"$version_without_v\"/" pyproject.toml
        fi
    fi
    
    # Update Dockerfile labels
    if [[ -f "Dockerfile" ]]; then
        log_info "Updating Dockerfile version label"
        if [[ "$DRY_RUN" == "false" ]]; then
            sed -i "s/LABEL version=\".*\"/LABEL version=\"$version_without_v\"/" Dockerfile || true
        fi
    fi
}

# Create git tag
create_git_tag() {
    local next_version="$1"
    
    log_info "Creating git tag: $next_version"
    
    # Commit version changes
    if [[ -n $(git status --porcelain) ]]; then
        execute "git add package.json pyproject.toml Dockerfile 2>/dev/null || true"
        execute "git commit -m \"chore: bump version to $next_version\""
    fi
    
    # Create annotated tag
    execute "git tag -a $next_version -m \"Release $next_version\""
    
    # Push changes and tag
    execute "git push origin $BRANCH"
    execute "git push origin $next_version"
    
    log_success "Git tag created and pushed: $next_version"
}

# Push to registry
push_to_registry() {
    if [[ "$SKIP_PUSH" == "true" ]]; then
        log_warning "Skipping push to registry as requested"
        return
    fi
    
    local next_version="$1"
    
    log_info "Pushing to Docker registry..."
    
    # Login to registry (assumes authentication is configured)
    log_info "Pushing $DOCKER_REGISTRY/$DOCKER_IMAGE:$next_version"
    execute "docker push $DOCKER_REGISTRY/$DOCKER_IMAGE:$next_version"
    execute "docker push $DOCKER_REGISTRY/$DOCKER_IMAGE:latest"
    
    log_success "Images pushed to registry"
}

# Create GitHub release
create_github_release() {
    local next_version="$1"
    
    if ! command -v gh >/dev/null 2>&1; then
        log_warning "GitHub CLI not found, skipping GitHub release creation"
        return
    fi
    
    log_info "Creating GitHub release..."
    
    # Generate release notes
    local previous_tag
    previous_tag=$(git describe --tags --abbrev=0 HEAD~1 2>/dev/null || echo "")
    
    local release_notes="Release $next_version\n\n"
    if [[ -n "$previous_tag" ]]; then
        release_notes+="## Changes since $previous_tag\n\n"
        release_notes+="$(git log --pretty=format:"- %s" "$previous_tag"..HEAD)\n\n"
    fi
    
    release_notes+="## Artifacts\n\n"
    release_notes+="- Docker Image: \`$DOCKER_REGISTRY/$DOCKER_IMAGE:$next_version\`\n"
    release_notes+="- SBOM: Available in release assets\n"
    
    # Create release
    if [[ -f "sbom-*.tar.gz" ]]; then
        execute "gh release create $next_version --title \"Release $next_version\" --notes \"$release_notes\" sbom-*.tar.gz"
    else
        execute "gh release create $next_version --title \"Release $next_version\" --notes \"$release_notes\""
    fi
    
    log_success "GitHub release created: $next_version"
}

# Display summary
show_summary() {
    local next_version="$1"
    
    echo ""
    log_info "Release Summary"
    echo "==============="
    echo "Version: $next_version"
    echo "Branch: $BRANCH"
    echo "Registry: $DOCKER_REGISTRY"
    echo "Image: $DOCKER_IMAGE"
    echo ""
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_warning "This was a dry run - no changes were made"
    else
        log_success "Release $next_version completed successfully!"
    fi
    
    echo ""
    log_info "Next steps:"
    echo "1. Verify the release on GitHub"
    echo "2. Test the Docker image: docker run $DOCKER_REGISTRY/$DOCKER_IMAGE:$next_version"
    echo "3. Update documentation if needed"
    echo "4. Announce the release to stakeholders"
}

# Main execution
main() {
    parse_args "$@"
    validate_args
    
    log_info "Starting release process..."
    log_info "Version type: $VERSION_TYPE"
    log_info "Branch: $BRANCH"
    log_info "Dry run: $DRY_RUN"
    
    check_prerequisites
    
    # Get current and next version
    current_version=$(get_current_version)
    next_version=$(calculate_next_version "$current_version")
    
    log_info "Current version: $current_version"
    log_info "Next version: $next_version"
    
    # Run release steps
    run_tests
    build_artifacts "$next_version"
    generate_sbom
    update_version_files "$next_version"
    create_git_tag "$next_version"
    push_to_registry "$next_version"
    create_github_release "$next_version"
    
    show_summary "$next_version"
}

# Run main function
main "$@"