#!/bin/bash
set -e

echo "🚀 Setting up AGI Evaluation Sandbox development environment..."

# Update package manager
sudo apt-get update

# Install system dependencies
sudo apt-get install -y \
    build-essential \
    curl \
    wget \
    git \
    vim \
    htop \
    tree \
    jq \
    postgresql-client \
    redis-tools

# Install Python dependencies
pip install --upgrade pip setuptools wheel

# Install development tools
pip install \
    black \
    isort \
    flake8 \
    mypy \
    pytest \
    pytest-cov \
    pytest-asyncio \
    pre-commit \
    jupyterlab \
    ipywidgets

# Install project dependencies (if requirements files exist)
if [ -f "requirements.txt" ]; then
    echo "📦 Installing project requirements..."
    pip install -r requirements.txt
fi

if [ -f "requirements-dev.txt" ]; then
    echo "🛠️ Installing development requirements..."
    pip install -r requirements-dev.txt
fi

# Install Node.js dependencies (if package.json exists)
if [ -f "package.json" ]; then
    echo "📦 Installing Node.js dependencies..."
    npm install
fi

# Setup pre-commit hooks
if [ -f ".pre-commit-config.yaml" ]; then
    echo "🔧 Setting up pre-commit hooks..."
    pre-commit install
fi

# Create necessary directories
mkdir -p \
    logs \
    data \
    results \
    .pytest_cache \
    .mypy_cache

# Set up Git configuration if not already configured
if [ -z "$(git config --global user.email)" ]; then
    echo "⚙️ Setting up Git configuration..."
    git config --global user.email "dev@agi-eval-sandbox.local"
    git config --global user.name "AGI Eval Dev"
fi

# Make scripts executable
find . -name "*.sh" -type f -exec chmod +x {} \;

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick start commands:"
echo "  make dev        # Start development services"
echo "  make test       # Run test suite"
echo "  make lint       # Run code quality checks"
echo "  make docs       # Build documentation"
echo ""
echo "📖 See README.md for detailed instructions"