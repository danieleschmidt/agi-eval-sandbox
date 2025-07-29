# CI/CD Workflow Setup Guide

This document provides the GitHub Actions workflows needed for comprehensive CI/CD automation.

## Required Workflows

### 1. Main CI/CD Pipeline (`.github/workflows/ci.yml`)

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test-api:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: postgres
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
          --health-interval 10s
          --health-timeout 5s
          --health-retries 5

    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        pip install -e ".[dev,test]"
    - name: Run tests
      run: |
        npm run test:api
    - name: Upload coverage
      uses: codecov/codecov-action@v3

  test-dashboard:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-node@v4
      with:
        node-version: '18'
    - name: Install dependencies
      run: npm install
    - name: Run tests
      run: npm run test:dashboard

  e2e-tests:
    runs-on: ubuntu-latest
    needs: [test-api, test-dashboard]
    steps:
    - uses: actions/checkout@v4
    - uses: actions/setup-node@v4
      with:
        node-version: '18'
    - name: Install Playwright
      run: npx playwright install --with-deps
    - name: Run E2E tests
      run: npm run test:e2e

  security-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Run security scan
      run: npm run security:scan
```

### 2. Security Scanning (`.github/workflows/security.yml`)

```yaml
name: Security Scan

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]
  schedule:
    - cron: '0 2 * * 1' # Weekly

jobs:
  dependency-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Python Security Scan
      uses: pypa/gh-action-pip-audit@v1.0.8
    - name: Node.js Security Scan
      run: npm audit --audit-level moderate

  code-analysis:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: CodeQL Analysis
      uses: github/codeql-action/init@v2
      with:
        languages: python, javascript
    - name: Perform CodeQL Analysis
      uses: github/codeql-action/analyze@v2

  container-scan:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build Docker image
      run: docker build -t agi-eval-sandbox .
    - name: Scan container
      uses: anchore/scan-action@v3
      with:
        image: agi-eval-sandbox
        fail-build: true
```

### 3. Performance Testing (`.github/workflows/performance.yml`)

```yaml
name: Performance Tests

on:
  push:
    branches: [ main ]
  pull_request:
    branches: [ main ]

jobs:
  lighthouse:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
    - name: Install dependencies
      run: npm install
    - name: Start services
      run: docker-compose up -d
    - name: Run Lighthouse
      uses: treosh/lighthouse-ci-action@v9
      with:
        configPath: ./lighthouse.config.js

  load-testing:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Start services
      run: docker-compose up -d
    - name: Run k6 load tests
      uses: grafana/k6-action@v0.3.0
      with:
        filename: performance/k6/load-test.js
```

### 4. Release Automation (`.github/workflows/release.yml`)

```yaml
name: Release

on:
  push:
    branches: [ main ]

jobs:
  release:
    runs-on: ubuntu-latest
    if: "!contains(github.event.head_commit.message, 'skip ci')"
    steps:
    - uses: actions/checkout@v4
      with:
        fetch-depth: 0
        token: ${{ secrets.GH_TOKEN }}
    
    - name: Setup Node.js
      uses: actions/setup-node@v4
      with:
        node-version: '18'
    
    - name: Install dependencies
      run: npm install
    
    - name: Build packages
      run: npm run build
    
    - name: Semantic Release
      env:
        GITHUB_TOKEN: ${{ secrets.GH_TOKEN }}
        NPM_TOKEN: ${{ secrets.NPM_TOKEN }}
      run: npx semantic-release

  docker-build:
    needs: release
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    - name: Build and push Docker image
      uses: docker/build-push-action@v5
      with:
        context: .
        push: true
        tags: |
          your-org/agi-eval-sandbox:latest
          your-org/agi-eval-sandbox:${{ github.sha }}
```

## Repository Secrets Required

Add these secrets in your GitHub repository settings:

- `GH_TOKEN`: GitHub token with repo permissions
- `NPM_TOKEN`: NPM token for package publishing
- `DOCKER_USERNAME`: Docker Hub username
- `DOCKER_PASSWORD`: Docker Hub password
- `CODECOV_TOKEN`: Codecov token for coverage reports

## Branch Protection Rules

Configure these branch protection rules for `main`:

- Require pull request reviews before merging
- Require status checks to pass before merging
- Require branches to be up to date before merging
- Include administrators in restrictions
- Required status checks:
  - `test-api`
  - `test-dashboard`
  - `e2e-tests`
  - `security-scan`

## Integration Setup

### SonarCloud Integration

Add to your workflow:

```yaml
- name: SonarCloud Scan
  uses: SonarSource/sonarcloud-github-action@master
  env:
    GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
    SONAR_TOKEN: ${{ secrets.SONAR_TOKEN }}
```

### Dependabot Configuration

Create `.github/dependabot.yml`:

```yaml
version: 2
updates:
  - package-ecosystem: "pip"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "npm"
    directory: "/"
    schedule:
      interval: "weekly"
  - package-ecosystem: "docker"
    directory: "/"
    schedule:
      interval: "weekly"
```

## Manual Setup Steps

1. Create the workflow files in `.github/workflows/`
2. Add required repository secrets
3. Configure branch protection rules
4. Enable Dependabot alerts and security updates
5. Set up SonarCloud project integration
6. Configure Codecov integration