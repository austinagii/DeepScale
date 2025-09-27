# Testing

DeepScale uses pytest for testing with comprehensive coverage.

## Running Tests

### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_config.py

# Run specific test
pytest tests/test_config.py::TestConfig::test_load_config
```

### Test Coverage

```bash
# Run with coverage
coverage run -m pytest
coverage report
coverage html  # Generate HTML report
```

### Verbose Output

```bash
# Verbose output
pytest -v

# Show stdout
pytest -s

# Stop on first failure
pytest -x
```

## Test Organization

Tests are organized by component:

```
tests/
├── test_config.py          # Configuration tests
├── test_training.py        # Training tests
├── storage/
│   ├── test_base.py        # Base storage tests
│   └── clients/
│       ├── test_file_system.py
│       └── test_azure_blob.py
└── fixtures/               # Test fixtures
```

## Writing Tests

### Basic Test Structure

```python
import pytest
from deepscale import Config

class TestConfig:
    def test_load_from_file(self):
        config = Config.from_file("test_config.yaml")
        assert config.training.epochs == 100

    def test_environment_override(self, monkeypatch):
        monkeypatch.setenv("DEEPSCALE_TRAINING_EPOCHS", "50")
        config = Config.from_file("test_config.yaml")
        assert config.training.epochs == 50
```

### Using Fixtures

```python
@pytest.fixture
def sample_config():
    return {
        "training": {"epochs": 10},
        "storage": {"backend": "file_system"}
    }

def test_with_fixture(sample_config):
    config = Config.from_dict(sample_config)
    assert config.training.epochs == 10
```

## Integration Tests

Run integration tests with external services:

```bash
# Skip integration tests
pytest -m "not integration"

# Run only integration tests
pytest -m integration
```