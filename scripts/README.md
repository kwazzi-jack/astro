# Test Scripts for Astro Project

This document describes the available test scripts for the Astro project. These scripts are only available when you have the development dependencies installed.

## Installation

First, install the development dependencies:

```bash
uv sync --group dev
```

## Available Test Scripts

All scripts use the `uv run` command:

### Basic Testing

- **`uv run test`** - Run all tests with verbose output
- **`uv run test-cov`** - Run all tests with coverage reporting (generates HTML and terminal reports)

### Category-Based Testing

- **`uv run test-unit`** - Run only unit tests (fast, isolated tests)
- **`uv run test-filesystem`** - Run tests that interact with the file system
- **`uv run test-integration`** - Run integration tests
- **`uv run test-fast`** - Run all tests except slow ones (great for development)
- **`uv run test-slow`** - Run only the slow tests

### Development Workflow

- **`uv run test-watch`** - Run tests with short traceback and stop on first failure (good for TDD)
- **`uv run test-debug`** - Run tests with detailed output and no capture (for debugging)
- **`uv run test-failed`** - Re-run only the tests that failed in the last run

### Code Quality Checks

- **`uv run test-ruff`** - Run ruff linting checks on test and script files
- **`uv run test-black`** - Run black code formatting checks on test and script files  
- **`uv run test-mypy`** - Run mypy type checking on test and script files
- **`uv run test-quality`** - Run all code quality checks with summary report
- **`uv run test-full`** - Run all tests + all code quality checks (comprehensive suite)

## Examples

```bash
# Quick development cycle - run fast tests only
uv run test-fast

# Full test suite with coverage
uv run test-cov

# Debug a failing test
uv run test-debug

# Focus on unit tests during development
uv run test-unit

# Run only the tests you just broke
uv run test-failed

# Code quality checks
uv run test-ruff          # Linting only
uv run test-black         # Formatting check only  
uv run test-mypy          # Type checking only
uv run test-quality       # All quality checks with summary

# Comprehensive testing (tests + quality)
uv run test-full          # Everything - best for CI/CD
```

## Test Markers

The tests use pytest markers for categorization:

- `@pytest.mark.unit` - Fast, isolated unit tests
- `@pytest.mark.filesystem` - Tests that create/modify files
- `@pytest.mark.integration` - Tests that combine multiple components
- `@pytest.mark.slow` - Tests that take longer to run

## Coverage Reports

When you run `uv run test-cov`, coverage reports are generated in two formats:

1. **Terminal output** - Shows missing line numbers
2. **HTML report** - Detailed visual report in `htmlcov/index.html`

Open the HTML report in your browser for detailed coverage analysis:

```bash
uv run test-cov
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

## Adding New Tests

1. Create test files in the `tests/` directory
2. Name them `test_*.py`
3. Use appropriate markers (`@pytest.mark.unit`, `@pytest.mark.filesystem`, etc.)
4. Use the fixtures provided in `tests/conftest.py`

Example:

```python
import pytest
from astro.paths import ModelFileStore
from tests.conftest import MockTraceableModel

@pytest.mark.unit
def test_my_feature(tmp_path, mock_model):
    """Test my new feature."""
    store = ModelFileStore(tmp_path, MockTraceableModel)
    store.add(mock_model)
    assert len(store) == 1
```

## Pytest Plugins Integration

The project includes these pytest plugins for code quality:

- **pytest-ruff** - Integrates ruff linting with pytest
- **pytest-black** - Integrates black formatting checks with pytest  
- **pytest-mypy** - Integrates mypy type checking with pytest

These plugins run as part of the test suite and can be configured through `pyproject.toml`.

### Configuration

Code quality tools are configured in `pyproject.toml`:

```toml
[tool.ruff]
line-length = 88
target-version = "py312"

[tool.black] 
line-length = 88
target-version = ["py312"]

[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
```

## CI/CD Integration

For continuous integration, use:

```bash
# Fast feedback - exclude slow tests
uv run test-fast

# Full verification with coverage and quality
uv run test-full

# Just code quality checks
uv run test-quality
```