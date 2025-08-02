# Manual Setup Required

## Overview

Due to GitHub App permission limitations, some setup tasks must be performed manually by repository maintainers. This document outlines all required manual actions to complete the SDLC implementation.

## Required Permissions

The following GitHub repository permissions are required:

- **Actions**: Write (for creating workflows)
- **Contents**: Write (for creating files and branches)
- **Issues**: Write (for creating issue templates)
- **Pull Requests**: Write (for creating PR templates)
- **Repository administration**: Write (for configuring branch protection)
- **Secrets**: Write (for configuring repository secrets)

## 1. GitHub Actions Workflows

### Step 1: Create Workflow Files

Create the following files in `.github/workflows/` using the templates from `docs/workflows/examples/`:

#### Required Workflows:

1. **`.github/workflows/ci.yml`**
   - Copy from: `docs/workflows/examples/ci.yml`
   - Purpose: Continuous integration with comprehensive testing

2. **`.github/workflows/security.yml`**
   - Copy from: `docs/workflows/examples/security.yml`
   - Purpose: Security scanning and vulnerability assessment

3. **`.github/workflows/deploy.yml`**
   - Copy from: `docs/workflows/examples/deploy.yml`
   - Purpose: Automated deployment with blue-green strategy

### Step 2: Configure Repository Secrets

Add the following secrets in **Settings > Secrets and variables > Actions**:

#### Required Secrets:
- `GITHUB_TOKEN` (automatically provided)

#### Optional Secrets for Enhanced Features:
- `SLACK_WEBHOOK_URL` - For Slack notifications
- `CODECOV_TOKEN` - For coverage reporting
- `SONAR_TOKEN` - For SonarCloud integration

## 2. Branch Protection Rules

### Step 1: Configure Main Branch Protection

Go to **Settings > Branches** and add protection rules for `main`:

```yaml
Branch Protection Rules for 'main':
  ✓ Require pull request reviews before merging
    - Required number of reviewers: 1
    - Dismiss stale reviews when new commits are pushed
    - Require review from code owners
  
  ✓ Require status checks to pass before merging
    - Require branches to be up to date before merging
    - Required status checks:
      - Code Quality
      - Python Tests
      - Build Application
      - Security Scan
  
  ✓ Require conversation resolution before merging
  ✓ Include administrators
  ✓ Allow force pushes (disable)
  ✓ Allow deletions (disable)
```

## 3. Repository Configuration

### Step 1: General Settings

Navigate to **Settings > General**:

```yaml
Repository Settings:
  - Description: "One-click evaluation environment for large language models"
  - Website: "https://agi-eval.com" (if applicable)
  - Topics: ["ai", "evaluation", "llm", "benchmarks", "machine-learning", "testing", "automation"]
  
  Features:
    ✓ Wikis (enabled)
    ✓ Issues (enabled) 
    ✓ Sponsorships (enabled if desired)
    ✓ Projects (enabled)
    ✓ Preserve this repository (enabled)
    ✓ Discussions (enabled if desired)
  
  Pull Requests:
    ✓ Allow merge commits
    ✓ Allow squash merging
    ✓ Allow rebase merging
    ✓ Always suggest updating pull request branches
    ✓ Automatically delete head branches
```

### Step 2: Security Settings

Navigate to **Settings > Security**:

```yaml
Security Settings:
  Code Security:
    ✓ Dependency graph (enabled)
    ✓ Dependabot alerts (enabled)
    ✓ Dependabot security updates (enabled)
    ✓ Dependabot version updates (enabled)
    ✓ Code scanning (enabled)
    ✓ Secret scanning (enabled)
    ✓ Secret scanning push protection (enabled)
```

## 4. Environment Configuration

### Step 1: Create Environments

Navigate to **Settings > Environments** and create:

#### Staging Environment:
```yaml
Name: staging
Protection Rules:
  - Required reviewers: 0
  - Wait timer: 0 minutes
  - Restrict deployments to protected branches: ✓
Environment Secrets:
  - AWS_ACCESS_KEY_ID (if using AWS)
  - DATABASE_URL
  - REDIS_URL
```

#### Production Environment:
```yaml
Name: production
Protection Rules:
  - Required reviewers: 1
  - Wait timer: 5 minutes
  - Restrict deployments to protected branches: ✓
  - Deployment branches: Selected branches (main, release/*)
Environment Secrets:
  - AWS_ACCESS_KEY_ID (if using AWS)
  - DATABASE_URL
  - REDIS_URL
```

## 5. Issue and PR Templates

### Step 1: Create Issue Templates

Create in `.github/ISSUE_TEMPLATE/`:

#### Bug Report (`bug_report.yml`):
```yaml
name: Bug Report
description: File a bug report
title: "[Bug]: "
labels: ["bug", "needs-triage"]
body:
  - type: textarea
    id: what-happened
    attributes:
      label: What happened?
      description: Also tell us, what did you expect to happen?
      placeholder: Tell us what you see!
    validations:
      required: true
```

### Step 2: Create Pull Request Template

Create `.github/pull_request_template.md`:

```markdown
## Summary

Brief description of the changes in this PR.

## Type of Change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing

- [ ] Unit tests pass
- [ ] Integration tests pass
- [ ] E2E tests pass
- [ ] Manual testing completed

## Checklist

- [ ] My code follows the style guidelines of this project
- [ ] I have performed a self-review of my own code
- [ ] I have commented my code, particularly in hard-to-understand areas
- [ ] I have made corresponding changes to the documentation
- [ ] My changes generate no new warnings
- [ ] I have added tests that prove my fix is effective or that my feature works
- [ ] New and existing unit tests pass locally with my changes
```

## Completion Checklist

- [ ] All GitHub Actions workflows created
- [ ] Repository secrets configured
- [ ] Branch protection rules set up
- [ ] Issue and PR templates created
- [ ] Security features enabled
- [ ] Environments configured
- [ ] Post-setup validation completed

## Documentation References

- [CI/CD Setup Guide](docs/workflows/CI_CD_SETUP.md)
- [SLSA Compliance Guide](docs/workflows/SLSA_COMPLIANCE.md)
- [Automation Guide](docs/automation/AUTOMATION_GUIDE.md)
- [Observability Guide](docs/monitoring/OBSERVABILITY_GUIDE.md)
- [Testing Strategy](docs/testing/TESTING_STRATEGY.md)