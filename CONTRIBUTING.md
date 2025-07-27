# Contributing to AGI Evaluation Sandbox

Thank you for your interest in contributing to the AGI Evaluation Sandbox! This document provides guidelines and information for contributors.

## 🎯 Ways to Contribute

- **🐛 Bug Reports**: Report issues or unexpected behavior
- **💡 Feature Requests**: Suggest new features or improvements
- **📝 Documentation**: Improve docs, examples, or tutorials
- **🔧 Code Contributions**: Fix bugs or implement features
- **🧪 Testing**: Add tests or improve test coverage
- **🎨 Design**: UI/UX improvements for the dashboard
- **📊 Benchmarks**: Add new evaluation benchmarks

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ with pip
- Node.js 18+ with npm
- Docker and Docker Compose
- Git

### Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/agi-eval-sandbox.git
   cd agi-eval-sandbox
   ```

2. **Environment Setup**
   ```bash
   # Copy environment template
   cp .env.example .env
   
   # Edit .env with your configuration
   nano .env
   ```

3. **Development Environment**
   ```bash
   # Option 1: Using Docker (Recommended)
   docker-compose -f docker-compose.dev.yml up
   
   # Option 2: Local Development
   make setup
   ```

4. **Verify Installation**
   ```bash
   make test
   make lint
   ```

### Development Workflow

1. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b bugfix/issue-number-description
   ```

2. **Make Changes**
   - Follow our coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

3. **Test Your Changes**
   ```bash
   # Run all tests
   make test
   
   # Run specific test categories
   make test-unit
   make test-integration
   make test-e2e
   
   # Check code quality
   make lint
   make typecheck
   ```

4. **Commit Changes**
   ```bash
   # We use conventional commits
   git add .
   git commit -m "feat: add new benchmark evaluation metric"
   # or
   git commit -m "fix: resolve memory leak in evaluation worker"
   ```

5. **Push and Create PR**
   ```bash
   git push origin feature/your-feature-name
   ```
   Then create a pull request on GitHub.

## 📋 Coding Standards

### Python Code Style

- **Formatter**: Black with 88 character line length
- **Import Sorting**: isort with black profile
- **Linting**: Flake8 with custom configuration
- **Type Hints**: Required for all functions and methods
- **Docstrings**: Google-style docstrings for public APIs

```python
def evaluate_model(
    model: ModelProvider,
    benchmark: Benchmark,
    config: EvaluationConfig
) -> EvaluationResult:
    """Evaluate a model on a specific benchmark.
    
    Args:
        model: The model provider instance to evaluate.
        benchmark: The benchmark to run evaluation on.
        config: Configuration parameters for evaluation.
        
    Returns:
        The evaluation results including metrics and outputs.
        
    Raises:
        EvaluationError: If evaluation fails due to model or benchmark issues.
    """
    # Implementation here
```

### JavaScript/TypeScript Code Style

- **Formatter**: Prettier with 2-space indentation
- **Linting**: ESLint with TypeScript rules
- **Framework**: React 18 with TypeScript
- **State Management**: Redux Toolkit (RTK Query for API calls)

### Git Commit Messages

We follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or modifying tests
- `chore`: Maintenance tasks

**Examples:**
- `feat(api): add support for custom evaluation metrics`
- `fix(dashboard): resolve chart rendering issue in Firefox`
- `docs(readme): update installation instructions`

## 🧪 Testing Guidelines

### Test Organization

```
tests/
├── unit/           # Fast, isolated unit tests
├── integration/    # Integration tests with external services
├── e2e/           # End-to-end tests with Playwright
└── fixtures/      # Test data and fixtures
```

### Writing Tests

- **Unit Tests**: Test individual functions/methods in isolation
- **Integration Tests**: Test component interactions
- **E2E Tests**: Test complete user workflows
- **Performance Tests**: Test for performance regressions

```python
# Example unit test
def test_evaluation_metrics_calculation():
    """Test that evaluation metrics are calculated correctly."""
    results = [
        {"correct": True, "confidence": 0.9},
        {"correct": False, "confidence": 0.7},
        {"correct": True, "confidence": 0.8},
    ]
    
    metrics = calculate_metrics(results)
    
    assert metrics["accuracy"] == 2/3
    assert metrics["avg_confidence"] == 0.8
```

### Test Coverage

- Maintain **>80% code coverage**
- All new features must include tests
- Bug fixes should include regression tests

## 📚 Documentation

### API Documentation

- Use OpenAPI/Swagger specifications
- Include request/response examples
- Document error conditions

### Code Documentation

- Public APIs require comprehensive docstrings
- Complex algorithms need inline comments
- Include type hints for better IDE support

### User Documentation

- Clear setup and usage instructions
- Working code examples
- Troubleshooting guides

## 🔄 Pull Request Process

### Before Submitting

- [ ] Tests pass locally (`make test`)
- [ ] Code follows style guidelines (`make lint`)
- [ ] Documentation updated if needed
- [ ] CHANGELOG.md updated for user-facing changes
- [ ] Self-review completed

### PR Requirements

1. **Clear Description**
   - What changes were made?
   - Why were they made?
   - How to test the changes?

2. **Linked Issues**
   - Reference related issues
   - Use "Closes #123" for bug fixes

3. **Breaking Changes**
   - Clearly document any breaking changes
   - Update migration guides if needed

### Review Process

1. **Automated Checks**: All CI checks must pass
2. **Code Review**: At least one maintainer review required
3. **Testing**: Manual testing for significant changes
4. **Approval**: Maintainer approval before merge

## 🏗️ Architecture Guidelines

### Adding New Benchmarks

1. **Implement Benchmark Interface**
   ```python
   class CustomBenchmark(BaseBenchmark):
       def load_questions(self) -> List[Question]:
           # Load benchmark questions
           
       def evaluate_response(self, question: Question, response: str) -> Score:
           # Evaluate model response
   ```

2. **Add Configuration Schema**
   ```python
   class CustomBenchmarkConfig(BenchmarkConfig):
       custom_param: str
       another_param: Optional[int] = None
   ```

3. **Register Benchmark**
   ```python
   @register_benchmark("custom-benchmark")
   class CustomBenchmark(BaseBenchmark):
       # Implementation
   ```

### Adding Model Providers

1. **Implement Provider Interface**
   ```python
   class CustomProvider(ModelProvider):
       async def generate(self, prompt: str, **kwargs) -> str:
           # Generate response
           
       def get_limits(self) -> RateLimits:
           # Return rate limits
   ```

2. **Add Provider Configuration**
   ```python
   class CustomProviderConfig(ProviderConfig):
       api_key: str
       base_url: Optional[str] = None
   ```

## 🐛 Bug Reports

### Good Bug Reports Include

1. **Clear Title**: Concise description of the issue
2. **Environment**: OS, Python version, package versions
3. **Steps to Reproduce**: Minimal reproducible example
4. **Expected Behavior**: What should happen
5. **Actual Behavior**: What actually happens
6. **Logs/Screenshots**: Relevant error messages or visuals

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**Environment**
- OS: [e.g., Ubuntu 20.04]
- Python: [e.g., 3.9.7]
- AGI Eval Sandbox: [e.g., 1.2.3]

**Steps to Reproduce**
1. Configure model with...
2. Run evaluation with...
3. See error...

**Expected Behavior**
Evaluation should complete successfully and return metrics.

**Actual Behavior**
Evaluation fails with timeout error.

**Logs**
```
[Include relevant log output]
```

**Additional Context**
Any other relevant information.
```

## 💡 Feature Requests

### Good Feature Requests Include

1. **Problem Statement**: What problem does this solve?
2. **Proposed Solution**: How should it work?
3. **Use Cases**: Who would use this feature?
4. **Implementation Ideas**: Technical approach (optional)

## 🤝 Community Guidelines

### Be Respectful

- Use inclusive language
- Be constructive in feedback
- Help others learn and grow
- Celebrate diversity of thought

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: General questions and ideas
- **Discord**: Real-time community chat
- **Email**: Private security or sensitive matters

## 📄 License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

## 🙏 Recognition

Contributors are recognized in:
- Release notes for significant contributions
- Contributors section in README
- Annual contributor highlights

## ❓ Questions?

- **General Questions**: GitHub Discussions
- **Bug Reports**: GitHub Issues
- **Security Issues**: security@terragon.ai
- **Private Matters**: contribute@terragon.ai

Thank you for contributing to AGI Evaluation Sandbox! 🚀