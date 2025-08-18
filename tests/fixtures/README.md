# Test Fixtures

This directory contains test data fixtures and mock objects used across the test suite.

## Directory Structure

```
fixtures/
├── README.md              # This file
├── data/                  # Test data files
│   ├── benchmarks/       # Sample benchmark data
│   ├── models/           # Mock model responses
│   ├── results/          # Sample evaluation results
│   └── configs/          # Test configuration files
├── mocks/                # Mock objects and factories
│   ├── __init__.py
│   ├── model_mocks.py    # Mock model providers
│   ├── api_mocks.py      # Mock API responses
│   └── service_mocks.py  # Mock service implementations
└── factories/            # Test data factories
    ├── __init__.py
    ├── benchmark_factory.py
    ├── model_factory.py
    └── result_factory.py
```

## Usage

### Test Data
Test data fixtures are organized by type and can be loaded using the fixture utilities:

```python
from tests.fixtures import load_benchmark_data, load_model_response

# Load sample benchmark questions
questions = load_benchmark_data("mmlu_sample.json")

# Load mock model response
response = load_model_response("gpt4_truthful_qa.json")
```

### Mock Objects
Use mock objects for testing components in isolation:

```python
from tests.fixtures.mocks import MockModelProvider

def test_evaluation_with_mock_model():
    mock_model = MockModelProvider("test-model")
    # Test evaluation logic without real API calls
```

### Factories
Use factories to generate test data with specific characteristics:

```python
from tests.fixtures.factories import BenchmarkFactory, ModelFactory

def test_evaluation_pipeline():
    benchmark = BenchmarkFactory.create(num_questions=10)
    model = ModelFactory.create(provider="mock")
    # Test with generated data
```

## Guidelines

1. **Deterministic Data**: All fixtures should produce deterministic results for reproducible tests
2. **Realistic Data**: Test data should closely resemble real-world usage patterns
3. **Minimal Data**: Keep fixture data as small as possible while maintaining realism
4. **Documentation**: Document any special characteristics or requirements for fixtures
5. **Versioning**: Include version information in fixture files for compatibility tracking