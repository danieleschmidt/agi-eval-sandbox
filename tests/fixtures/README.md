# Test Fixtures

This directory contains test data and fixtures used across the test suite.

## Structure

- `data/` - Static test data files (JSON, CSV, etc.)
- `models/` - Test model configurations and mock responses
- `benchmarks/` - Sample benchmark data for testing
- `factories/` - Test data factories for dynamic test data generation

## Usage

### Loading Fixtures

```python
import pytest
from pathlib import Path

@pytest.fixture
def sample_evaluation_data():
    fixtures_dir = Path(__file__).parent / "fixtures"
    with open(fixtures_dir / "data" / "sample_evaluation.json") as f:
        return json.load(f)
```

### Factory Patterns

```python
from tests.fixtures.factories import ModelFactory, BenchmarkFactory

def test_evaluation():
    model = ModelFactory.create(name="test-model")
    benchmark = BenchmarkFactory.create(type="multiple_choice")
    # Test logic here
```

## Guidelines

1. Keep fixtures small and focused
2. Use factories for dynamic data generation
3. Store static data in appropriate subdirectories
4. Document complex fixture relationships
5. Use descriptive file names that indicate their purpose