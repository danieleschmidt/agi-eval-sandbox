# AGI Evaluation Sandbox - Makefile
# =====================================

.PHONY: help setup clean dev test lint format typecheck build docker security docs

# Default target
.DEFAULT_GOAL := help

# Colors for output
BLUE := \033[36m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
RESET := \033[0m

# Project info
PROJECT_NAME := agi-eval-sandbox
PYTHON_VERSION := 3.11
NODE_VERSION := 18

help: ## Show this help message
	@echo "$(BLUE)$(PROJECT_NAME) - Development Commands$(RESET)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(RESET) %s\n", $$1, $$2}'

# ===========================================
# Environment Setup
# ===========================================

setup: ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(RESET)"
	@if [ ! -f .env ]; then \
		echo "$(YELLOW)Creating .env from .env.example...$(RESET)"; \
		cp .env.example .env; \
	fi
	@if command -v python3 >/dev/null 2>&1; then \
		echo "$(GREEN)Python found, setting up virtual environment...$(RESET)"; \
		python3 -m venv .venv; \
		. .venv/bin/activate && pip install --upgrade pip setuptools wheel; \
	fi
	@if [ -f requirements.txt ]; then \
		echo "$(GREEN)Installing Python dependencies...$(RESET)"; \
		. .venv/bin/activate && pip install -r requirements.txt; \
	fi
	@if [ -f requirements-dev.txt ]; then \
		echo "$(GREEN)Installing development dependencies...$(RESET)"; \
		. .venv/bin/activate && pip install -r requirements-dev.txt; \
	fi
	@if command -v npm >/dev/null 2>&1; then \
		echo "$(GREEN)Installing Node.js dependencies...$(RESET)"; \
		npm install; \
	fi
	@if [ -f .pre-commit-config.yaml ]; then \
		echo "$(GREEN)Setting up pre-commit hooks...$(RESET)"; \
		. .venv/bin/activate && pre-commit install; \
	fi
	@echo "$(GREEN)Setup complete! Run 'make dev' to start development servers.$(RESET)"

clean: ## Clean up build artifacts and caches
	@echo "$(BLUE)Cleaning up...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf node_modules/
	rm -rf .next/
	rm -rf coverage/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "$(GREEN)Cleanup complete!$(RESET)"

# ===========================================
# Development
# ===========================================

dev: ## Start development servers
	@echo "$(BLUE)Starting development servers...$(RESET)"
	@if [ -f docker-compose.dev.yml ]; then \
		docker-compose -f docker-compose.dev.yml up --build; \
	else \
		npm run dev; \
	fi

dev-api: ## Start only the API server
	@echo "$(BLUE)Starting API server...$(RESET)"
	cd api && . ../.venv/bin/activate && uvicorn main:app --reload --host 0.0.0.0 --port 8000

dev-dashboard: ## Start only the dashboard
	@echo "$(BLUE)Starting dashboard...$(RESET)"
	cd dashboard && npm run dev

dev-jupyter: ## Start Jupyter Lab
	@echo "$(BLUE)Starting Jupyter Lab...$(RESET)"
	. .venv/bin/activate && jupyter lab --ip=0.0.0.0 --port=8888 --no-browser

# ===========================================
# Testing
# ===========================================

test: ## Run all tests
	@echo "$(BLUE)Running all tests...$(RESET)"
	$(MAKE) test-api
	$(MAKE) test-dashboard

test-api: ## Run Python tests
	@echo "$(BLUE)Running Python tests...$(RESET)"
	@if [ -d api ]; then \
		cd api && . ../.venv/bin/activate && python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term; \
	else \
		. .venv/bin/activate && python -m pytest tests/ -v --cov=src --cov-report=html --cov-report=term; \
	fi

test-dashboard: ## Run JavaScript/TypeScript tests
	@echo "$(BLUE)Running dashboard tests...$(RESET)"
	@if [ -d dashboard ]; then \
		cd dashboard && npm test; \
	fi

test-e2e: ## Run end-to-end tests
	@echo "$(BLUE)Running E2E tests...$(RESET)"
	npx playwright test

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(RESET)"
	@if [ -d api ]; then \
		cd api && . ../.venv/bin/activate && python -m pytest tests/ -v --cov=src -f; \
	else \
		. .venv/bin/activate && python -m pytest tests/ -v --cov=src -f; \
	fi

# ===========================================
# Code Quality
# ===========================================

lint: ## Run all linters
	@echo "$(BLUE)Running linters...$(RESET)"
	$(MAKE) lint-api
	$(MAKE) lint-dashboard

lint-api: ## Run Python linting
	@echo "$(BLUE)Running Python linting...$(RESET)"
	@if [ -d api ]; then \
		cd api && . ../.venv/bin/activate && python -m flake8 src tests; \
	else \
		. .venv/bin/activate && python -m flake8 src tests; \
	fi

lint-dashboard: ## Run JavaScript/TypeScript linting
	@echo "$(BLUE)Running dashboard linting...$(RESET)"
	@if [ -d dashboard ]; then \
		cd dashboard && npm run lint; \
	fi

format: ## Format all code
	@echo "$(BLUE)Formatting code...$(RESET)"
	$(MAKE) format-api
	$(MAKE) format-dashboard

format-api: ## Format Python code
	@echo "$(BLUE)Formatting Python code...$(RESET)"
	@if [ -d api ]; then \
		cd api && . ../.venv/bin/activate && python -m black src tests && python -m isort src tests; \
	else \
		. .venv/bin/activate && python -m black src tests && python -m isort src tests; \
	fi

format-dashboard: ## Format JavaScript/TypeScript code
	@echo "$(BLUE)Formatting dashboard code...$(RESET)"
	@if [ -d dashboard ]; then \
		cd dashboard && npm run format; \
	fi

typecheck: ## Run type checking
	@echo "$(BLUE)Running type checking...$(RESET)"
	$(MAKE) typecheck-api
	$(MAKE) typecheck-dashboard

typecheck-api: ## Run Python type checking
	@echo "$(BLUE)Running Python type checking...$(RESET)"
	@if [ -d api ]; then \
		cd api && . ../.venv/bin/activate && python -m mypy src tests; \
	else \
		. .venv/bin/activate && python -m mypy src tests; \
	fi

typecheck-dashboard: ## Run TypeScript checking
	@echo "$(BLUE)Running TypeScript checking...$(RESET)"
	@if [ -d dashboard ]; then \
		cd dashboard && npm run typecheck; \
	fi

pre-commit: ## Run pre-commit hooks on all files
	@echo "$(BLUE)Running pre-commit hooks...$(RESET)"
	. .venv/bin/activate && pre-commit run --all-files

# ===========================================
# Build & Package
# ===========================================

build: ## Build all components
	@echo "$(BLUE)Building all components...$(RESET)"
	$(MAKE) build-api
	$(MAKE) build-dashboard

build-api: ## Build Python package
	@echo "$(BLUE)Building Python package...$(RESET)"
	@if [ -d api ]; then \
		cd api && . ../.venv/bin/activate && python -m build; \
	else \
		. .venv/bin/activate && python -m build; \
	fi

build-dashboard: ## Build dashboard
	@echo "$(BLUE)Building dashboard...$(RESET)"
	@if [ -d dashboard ]; then \
		cd dashboard && npm run build; \
	fi

# ===========================================
# Docker
# ===========================================

docker-build: ## Build Docker images
	@echo "$(BLUE)Building Docker images...$(RESET)"
	docker build -t $(PROJECT_NAME):latest .

docker-run: ## Run with Docker Compose
	@echo "$(BLUE)Running with Docker Compose...$(RESET)"
	docker-compose up --build

docker-test: ## Run tests in Docker
	@echo "$(BLUE)Running tests in Docker...$(RESET)"
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

docker-clean: ## Clean Docker resources
	@echo "$(BLUE)Cleaning Docker resources...$(RESET)"
	docker system prune -f
	docker volume prune -f

# ===========================================
# Database
# ===========================================

db-migrate: ## Run database migrations
	@echo "$(BLUE)Running database migrations...$(RESET)"
	@if [ -d api ]; then \
		cd api && . ../.venv/bin/activate && alembic upgrade head; \
	fi

db-seed: ## Seed database with sample data
	@echo "$(BLUE)Seeding database...$(RESET)"
	@if [ -d api ]; then \
		cd api && . ../.venv/bin/activate && python scripts/seed_database.py; \
	fi

db-reset: ## Reset database (WARNING: destroys data)
	@echo "$(RED)WARNING: This will destroy all data!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo ""; \
		echo "$(BLUE)Resetting database...$(RESET)"; \
		if [ -d api ]; then \
			cd api && . ../.venv/bin/activate && alembic downgrade base && alembic upgrade head; \
		fi; \
	fi

# ===========================================
# Security
# ===========================================

security: ## Run security scans
	@echo "$(BLUE)Running security scans...$(RESET)"
	$(MAKE) security-python
	$(MAKE) security-node
	$(MAKE) security-docker

security-python: ## Run Python security scan
	@echo "$(BLUE)Running Python security scan...$(RESET)"
	. .venv/bin/activate && safety check
	. .venv/bin/activate && bandit -r src/ -f json -o security-report.json || true

security-node: ## Run Node.js security scan
	@echo "$(BLUE)Running Node.js security audit...$(RESET)"
	npm audit --audit-level moderate

security-docker: ## Run Docker security scan
	@echo "$(BLUE)Running Docker security scan...$(RESET)"
	@if command -v trivy >/dev/null 2>&1; then \
		trivy image $(PROJECT_NAME):latest; \
	else \
		echo "$(YELLOW)Trivy not found. Install with: brew install aquasecurity/trivy/trivy$(RESET)"; \
	fi

# ===========================================
# Documentation
# ===========================================

docs: ## Serve documentation locally
	@echo "$(BLUE)Starting documentation server...$(RESET)"
	@if [ -f docs/mkdocs.yml ]; then \
		cd docs && mkdocs serve; \
	else \
		echo "$(YELLOW)Documentation not found. Run 'make docs-init' to set up.$(RESET)"; \
	fi

docs-build: ## Build documentation
	@echo "$(BLUE)Building documentation...$(RESET)"
	@if [ -f docs/mkdocs.yml ]; then \
		cd docs && mkdocs build; \
	fi

docs-deploy: ## Deploy documentation to GitHub Pages
	@echo "$(BLUE)Deploying documentation...$(RESET)"
	@if [ -f docs/mkdocs.yml ]; then \
		cd docs && mkdocs gh-deploy; \
	fi

# ===========================================
# Benchmarks & Evaluation
# ===========================================

benchmark: ## Run sample benchmark
	@echo "$(BLUE)Running sample benchmark...$(RESET)"
	@if [ -d api ]; then \
		cd api && . ../.venv/bin/activate && python scripts/run_benchmark.py; \
	fi

evaluate: ## Run model evaluation (requires MODEL env var)
	@echo "$(BLUE)Running model evaluation...$(RESET)"
	@if [ -z "$(MODEL)" ]; then \
		echo "$(RED)Error: MODEL environment variable not set$(RESET)"; \
		echo "Usage: make evaluate MODEL=gpt-4"; \
		exit 1; \
	fi
	@if [ -d api ]; then \
		cd api && . ../.venv/bin/activate && python -m agi_eval_sandbox evaluate --model $(MODEL); \
	fi

# ===========================================
# Release & Deployment
# ===========================================

release: ## Create a new release (requires VERSION)
	@echo "$(BLUE)Creating release...$(RESET)"
	@if [ -z "$(VERSION)" ]; then \
		echo "$(RED)Error: VERSION not specified$(RESET)"; \
		echo "Usage: make release VERSION=1.0.0"; \
		exit 1; \
	fi
	git tag -a v$(VERSION) -m "Release v$(VERSION)"
	git push origin v$(VERSION)
	@echo "$(GREEN)Release v$(VERSION) created!$(RESET)"

deploy-staging: ## Deploy to staging environment
	@echo "$(BLUE)Deploying to staging...$(RESET)"
	# Add your staging deployment commands here

deploy-prod: ## Deploy to production environment
	@echo "$(BLUE)Deploying to production...$(RESET)"
	# Add your production deployment commands here

# ===========================================
# Monitoring
# ===========================================

logs: ## View application logs
	@echo "$(BLUE)Viewing logs...$(RESET)"
	docker-compose logs -f

status: ## Check service status
	@echo "$(BLUE)Checking service status...$(RESET)"
	docker-compose ps

# ===========================================
# Maintenance
# ===========================================

update-deps: ## Update all dependencies
	@echo "$(BLUE)Updating dependencies...$(RESET)"
	. .venv/bin/activate && pip-review --auto
	npm update
	@echo "$(GREEN)Dependencies updated!$(RESET)"

check-deps: ## Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(RESET)"
	. .venv/bin/activate && pip list --outdated
	npm outdated

install-tools: ## Install development tools
	@echo "$(BLUE)Installing development tools...$(RESET)"
	# Python tools
	pip install --upgrade pip-tools pip-review safety bandit
	# Node.js tools
	npm install -g npm-check-updates
	# Other tools (optional)
	@if command -v brew >/dev/null 2>&1; then \
		echo "$(GREEN)Installing additional tools via Homebrew...$(RESET)"; \
		brew install trivy hadolint; \
	fi

# ===========================================
# Environment Info
# ===========================================

info: ## Show environment information
	@echo "$(BLUE)Environment Information$(RESET)"
	@echo "========================"
	@echo "Project: $(PROJECT_NAME)"
	@echo "Python: $(shell python3 --version 2>/dev/null || echo 'Not found')"
	@echo "Node.js: $(shell node --version 2>/dev/null || echo 'Not found')"
	@echo "Docker: $(shell docker --version 2>/dev/null || echo 'Not found')"
	@echo "Git: $(shell git --version 2>/dev/null || echo 'Not found')"
	@echo ""
	@echo "$(BLUE)Current Git Branch:$(RESET) $(shell git branch --show-current 2>/dev/null || echo 'Unknown')"
	@echo "$(BLUE)Last Commit:$(RESET) $(shell git log -1 --pretty=format:'%h - %s (%an, %ar)' 2>/dev/null || echo 'No commits')"