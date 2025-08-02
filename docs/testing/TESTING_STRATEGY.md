# Testing Strategy

## Overview

The AGI Evaluation Sandbox employs a comprehensive testing strategy with multiple test types and frameworks to ensure reliability, performance, and security across all components.

## Test Pyramid

```
        /\
       /  \
      /    \
     / E2E  \     <- High-level integration tests
    /        \
   /          \
  / Integration \  <- API and service integration tests
 /              \
/      Unit       \ <- Fast, isolated component tests
------------------
```

## Test Types

### 1. Unit Tests
- **Purpose**: Test individual components/functions in isolation
- **Framework**: Jest (Frontend), pytest (Backend)
- **Coverage Target**: 80%+
- **Execution Time**: < 10 seconds total
- **Location**: `tests/unit/`

**Examples:**
- Component rendering
- Utility function behavior
- Business logic validation
- Error handling

### 2. Integration Tests
- **Purpose**: Test component interactions and API endpoints
- **Framework**: pytest with test client, Jest with MSW
- **Coverage Target**: 70%+
- **Execution Time**: < 30 seconds total
- **Location**: `tests/integration/`

**Examples:**
- API endpoint functionality
- Database operations
- Service communication
- External API mocking

### 3. End-to-End Tests
- **Purpose**: Test complete user workflows
- **Framework**: Playwright
- **Coverage Target**: Critical user paths
- **Execution Time**: < 5 minutes total
- **Location**: `tests/e2e/`

**Examples:**
- User authentication flow
- Evaluation creation and execution
- Dashboard navigation
- Report generation

### 4. Performance Tests
- **Purpose**: Validate system performance under load
- **Framework**: k6, Artillery
- **Metrics**: Response time, throughput, resource usage
- **Location**: `performance/`

**Examples:**
- API load testing
- Database query performance
- Concurrent evaluation handling
- Memory usage patterns

## Test Organization

### Directory Structure

```
tests/
├── conftest.py                 # pytest configuration and fixtures
├── setup/
│   └── jest.setup.ts          # Jest setup and global mocks
├── fixtures/
│   ├── data/                  # Static test data
│   ├── factories.py           # Dynamic test data generation
│   └── README.md             # Fixture usage guide
├── unit/
│   ├── backend/
│   │   ├── test_models.py
│   │   ├── test_services.py
│   │   └── test_utils.py
│   └── frontend/
│       ├── components/
│       ├── hooks/
│       └── utils/
├── integration/
│   ├── test_api_endpoints.py
│   ├── test_database.py
│   └── test_external_services.py
├── e2e/
│   ├── global-setup.ts
│   ├── global-teardown.ts
│   ├── auth.spec.ts
│   ├── evaluation.spec.ts
│   └── dashboard.spec.ts
└── __mocks__/
    └── fileMock.js
```

### Test Naming Conventions

- **Files**: `test_*.py` or `*.test.ts`
- **Classes**: `TestClassName` (Python), `describe('ComponentName')` (JS)
- **Methods**: `test_should_do_something` (Python), `it('should do something')` (JS)

## Test Configuration

### Python Tests (pytest)

```ini
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
addopts = 
    --strict-markers
    --cov=src
    --cov-report=html
    --cov-fail-under=80
```

### JavaScript Tests (Jest)

```javascript
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'jsdom',
  collectCoverageFrom: ['src/**/*.{ts,tsx}'],
  coverageThreshold: {
    global: { lines: 70 }
  }
};
```

### E2E Tests (Playwright)

```typescript
// playwright.config.ts
export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  retries: process.env.CI ? 2 : 0,
  use: {
    baseURL: 'http://localhost:8080',
    trace: 'on-first-retry'
  }
});
```

## Test Data Management

### Fixtures and Factories

```python
# tests/fixtures/factories.py
class ModelFactory(factory.Factory):
    class Meta:
        model = dict
    
    name = factory.Sequence(lambda n: f"test-model-{n}")
    provider = fuzzy.FuzzyChoice(["openai", "anthropic"])
```

### Database State

- Use transactions for test isolation
- Reset database between test suites
- Seed minimal required data
- Clean up test data after runs

## Mocking Strategy

### External Services

```python
# Mock external API calls
@pytest.fixture
def mock_openai_client():
    with patch('openai.OpenAI') as mock:
        mock.return_value.chat.completions.create.return_value = MockResponse()
        yield mock
```

### Database Operations

```python
# Use test database
@pytest.fixture(scope="function")
def test_db():
    # Setup test database
    yield db
    # Cleanup
```

## Continuous Integration

### Test Execution Pipeline

1. **Lint and Format Check**
   ```bash
   npm run lint
   npm run format:check
   ```

2. **Unit Tests**
   ```bash
   npm run test:unit
   pytest tests/unit/ -v
   ```

3. **Integration Tests**
   ```bash
   npm run test:integration
   pytest tests/integration/ -v
   ```

4. **E2E Tests**
   ```bash
   npm run test:e2e
   ```

5. **Performance Tests**
   ```bash
   npm run test:performance
   ```

### Test Parallelization

- Unit tests: Run in parallel by default
- Integration tests: Limited parallelization due to shared resources
- E2E tests: Browser-level parallelization
- Performance tests: Sequential execution

## Quality Gates

### Coverage Requirements

| Test Type | Minimum Coverage | Target Coverage |
|-----------|------------------|-----------------|
| Unit      | 80%              | 90%             |
| Integration| 70%             | 80%             |
| E2E       | Critical paths   | All user flows  |

### Performance Thresholds

| Metric | Threshold |
|--------|-----------|
| API Response Time | < 500ms |
| Page Load Time | < 2s |
| Test Suite Execution | < 5min |

## Test Environment Setup

### Local Development

```bash
# Install dependencies
npm install
pip install -e ".[test]"

# Setup test database
npm run db:test:setup

# Run all tests
npm run test
```

### CI/CD Environment

```yaml
# .github/workflows/test.yml
- name: Run Tests
  run: |
    npm run test:ci
    pytest --junitxml=test-results/pytest.xml
```

## Best Practices

### Writing Tests

1. **Arrange-Act-Assert**: Structure tests clearly
2. **One assertion per test**: Keep tests focused
3. **Descriptive names**: Make test intent clear
4. **Independent tests**: No test dependencies
5. **Fast execution**: Optimize for speed

### Test Maintenance

1. **Regular cleanup**: Remove obsolete tests
2. **Update fixtures**: Keep test data current
3. **Review coverage**: Identify gaps
4. **Performance monitoring**: Track test execution time
5. **Documentation**: Keep test docs updated

### Debugging Tests

1. **Verbose output**: Use `-v` flag for detailed results
2. **Selective execution**: Run specific test files/methods
3. **Debug mode**: Use debugger breakpoints
4. **Log analysis**: Check application logs
5. **Browser tools**: Use Playwright inspector for E2E tests

## Troubleshooting

### Common Issues

1. **Flaky tests**: Investigate timing issues, add waits
2. **Slow tests**: Profile and optimize database queries
3. **Memory leaks**: Check for proper cleanup
4. **CI failures**: Verify environment consistency
5. **Coverage drops**: Identify uncovered code paths

### Tools and Commands

```bash
# Run specific test file
pytest tests/unit/test_models.py
npm test -- ComponentName.test.ts

# Run with coverage
pytest --cov=src tests/
npm test -- --coverage

# Debug mode
pytest --pdb tests/unit/test_models.py
npm test -- --watchAll

# Performance profiling
pytest --profile tests/integration/
npm test -- --logHeapUsage
```

This testing strategy ensures comprehensive coverage while maintaining development velocity and system reliability.