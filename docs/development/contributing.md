# Contributing

Thank you for your interest in contributing to DeepScale!

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/deepscale.git
   cd deepscale
   ```
3. Install development dependencies:
   ```bash
   pipenv install --dev
   pipenv shell
   ```

## Code Style

DeepScale uses Ruff for linting and formatting:

```bash
# Format code
ruff format .

# Check for issues
ruff check .
```

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
coverage run -m pytest
coverage report
```

## Documentation

Build documentation locally:

```bash
# Serve documentation
mkdocs serve

# Build documentation
mkdocs build
```

## Pull Request Process

1. Create a feature branch from `main`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass
5. Update documentation if needed
6. Submit a pull request

## Code Review

All submissions require review. Please:

- Write clear commit messages
- Keep changes focused and atomic
- Add tests for new features
- Update documentation as needed