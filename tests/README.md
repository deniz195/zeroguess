# ZeroGuess Test Suite

This directory contains the test suite for the ZeroGuess library. The tests are organized according to the test strategy outlined in [tests.md](../tests.md).

## Directory Structure

- `unit/`: Unit tests for individual components
- `integration/`: Integration tests between components
- `end_to_end/`: End-to-end tests for complete workflows
- `edge_case/`: Tests for specific edge cases and failure modes
- `conftest/`: Common test fixtures and configuration

## Requirements

To run the tests, you'll need:

1. Python 3.10 or higher
2. Install test dependencies:

```bash
pip install pytest pytest-cov pytest-benchmark pytest-xdist numpy scipy
```

3. Optional: PyTorch for neural network tests:

```bash
pip install torch torchvision
```

## Running Tests

### Running All Tests

```bash
# From the project root
pytest
```

### Running Specific Test Types

```bash
# Unit tests only
pytest tests/unit/

# Integration tests only
pytest tests/integration/

# End-to-end tests only
pytest tests/end_to_end/

# Edge case tests only
pytest tests/edge_case/
```

### Running Tests with Coverage

```bash
pytest --cov=zeroguess
```

### Running Tests in Parallel

```bash
pytest -xvs -n auto
```

### Skipping Slow Tests

```bash
pytest -m "not slow"
```

## Test Markers

The following markers are available:

- `slow`: Marks tests that take a long time to run
- `integration`: Marks tests that test component interactions
- `end_to_end`: Marks tests that test the complete workflow
- `edge_case`: Marks tests for edge cases and failure modes

## Handling Non-Deterministic Behavior

The test suite uses a fixed random seed strategy to ensure reproducibility despite the non-deterministic nature of ML training. This is implemented through the `set_random_seeds` fixture in `conftest.py`.

## CI/CD Integration

The test suite is designed to integrate with CI/CD pipelines:

1. Fast tests run on every commit
2. Full tests run on PR merge
3. Extended tests (including statistical validations) run nightly

## Creating New Tests

When creating new tests:

1. Use the appropriate directory based on test type
2. Import fixtures from `conftest.py`
3. Use the `set_random_seeds` fixture for reproducibility
4. Follow the common test structure:
   - Setup
   - Training (if applicable)
   - Prediction
   - Validation
   - Integration verification (optional)

## Checking Test Coverage

To generate a test coverage report:

```bash
pytest --cov=zeroguess --cov-report=html
```

Then open `htmlcov/index.html` in your browser to view the coverage report. 