# Contributing to AGI Evaluation Sandbox

Thank you for your interest in contributing to the AGI Evaluation Sandbox! This document provides guidelines for contributing to the project.

## Code of Conduct

This project adheres to a code of conduct. By participating, you are expected to uphold this code.

## How to Contribute

### Reporting Issues
- Use GitHub Issues to report bugs or request features
- Search existing issues before creating new ones
- Provide detailed reproduction steps for bugs
- Include system information and error messages

### Submitting Code Changes

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make your changes**
4. **Run tests**: `make test`
5. **Commit using conventional commits**: `git commit -m "feat: add new feature"`
6. **Push to your fork**: `git push origin feature/your-feature`
7. **Create a Pull Request**

### Pull Request Guidelines

- Fill out the PR template completely
- Ensure all tests pass
- Include tests for new functionality
- Update documentation as needed
- Follow the coding standards
- Keep changes focused and atomic

## Development Setup

See [DEVELOPMENT.md](docs/DEVELOPMENT.md) for detailed setup instructions.

## Coding Standards

- Follow PEP 8 for Python code
- Use Prettier for JavaScript/TypeScript
- Include type hints for Python functions
- Write comprehensive docstrings
- Maintain test coverage above 80%

## Commit Convention

We use Conventional Commits:

- `feat:` New features
- `fix:` Bug fixes
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build system or dependency changes

## Release Process

1. Version bump follows semantic versioning
2. Changelog is automatically generated
3. Releases are tagged and published
4. Docker images are built and pushed

## Getting Help

- Join our [Discord community](https://discord.gg/agi-eval)
- Check the [documentation](https://docs.your-org.com/agi-eval)
- Email: support@your-org.com