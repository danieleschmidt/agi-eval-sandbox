# Development Guide

## Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-org/agi-eval-sandbox
   cd agi-eval-sandbox
   ```

2. **Setup development environment**
   ```bash
   make setup
   ```

3. **Start development servers**
   ```bash
   make dev
   ```

## Development Workflow

1. Create feature branch: `git checkout -b feature/your-feature`
2. Make changes and commit: `git commit -am "feat: your feature"`
3. Run tests: `make test`
4. Push and create PR: `git push origin feature/your-feature`

## Architecture Overview

The AGI Evaluation Sandbox follows a microservices architecture:

- **API Gateway**: FastAPI-based REST API
- **Evaluation Engine**: Async task processing with Celery
- **Database**: PostgreSQL with TimescaleDB
- **Cache**: Redis for session and result caching
- **Frontend**: React-based dashboard

## Testing Strategy

- **Unit Tests**: Individual component testing
- **Integration Tests**: API and database integration
- **E2E Tests**: Full user workflow testing
- **Performance Tests**: Load and stress testing

Run all tests: `make test`

## Code Quality

- **Linting**: `make lint`
- **Formatting**: `make format`
- **Type Checking**: `make typecheck`
- **Security Scan**: `make security`

## Docker Development

- **Build**: `make docker-build`
- **Run**: `make docker-run`
- **Test**: `make docker-test`

## Contributing

Please read [CONTRIBUTING.md](../CONTRIBUTING.md) for detailed contribution guidelines.