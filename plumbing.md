# Plumbing

## Overview
This file describes the ZeroGuess project's coding and linting standards, CI/CD pipeline, Python package details, and development workflow.

## Code Quality Standards

### Formatting
- **Black**: We use Black for code formatting with a line length of 88 characters.
- **isort**: We use isort for sorting imports, configured to be compatible with Black.

### Linting
- **Flake8**: We use Flake8 for linting with the following plugins:
  - flake8-bugbear: For detecting potential bugs
  - flake8-docstrings: For docstring style checking
  - flake8-import-order: For import order checking
  - pep8-naming: For PEP 8 naming convention checking

### Type Checking
- **mypy**: We use mypy for static type checking with strict mode enabled.

### Unused Code Detection
- Flake8 with the built-in F401 (unused imports) and F841 (unused variables) checks
- Additionally, vulture is used for more comprehensive unused code detection

## Development Workflow

### Local Development
1. Install development dependencies: `pip install -e ".[dev]"`
2. Install pre-commit hooks: `pre-commit install`
3. Run tests: `pytest`
4. Run linting: `flake8`
5. Run type checking: `mypy zeroguess`

### Pre-commit Hooks
We use pre-commit hooks to automatically check code quality before commits:
- black: Formats code
- isort: Sorts imports
- flake8: Lints code
- mypy: Checks types

Install with: `pre-commit install`

## Running Tools Manually

### Running Tests
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=zeroguess

# Run specific test file
pytest tests/test_specific.py

# Run tests with specific marker
pytest -m "not slow"
```

### Running Linters
```bash
# Run flake8
flake8 zeroguess tests

# Run mypy
mypy zeroguess
```

### Running Formatters
```bash
# Run black
black zeroguess tests

# Run isort
isort zeroguess tests
```

## CI/CD Pipeline

### GitHub Actions
We use GitHub Actions for continuous integration with the following workflows:

1. **Test Workflow** (.github/workflows/test.yml):
   - Triggered on push to main, pull requests, and nightly schedule
   - Runs tests with pytest
   - Generates coverage reports
   - Uploads coverage to Codecov
   - Archives test visualization artifacts

2. **Lint Workflow** (.github/workflows/lint.yml) - *To be implemented*:
   - Triggered on push to main and pull requests
   - Runs black, isort, flake8, and mypy
   - Fails if any tool reports issues

3. **Release Workflow** (.github/workflows/release.yml) - *To be implemented*:
   - Triggered on release creation
   - Builds package
   - Publishes to PyPI

### Test Coverage
- We use pytest-cov to generate coverage reports
- Coverage reports are uploaded to Codecov
- Coverage badge is displayed in the README.md
- Target coverage: 80% or higher

## Package Management

### Dependencies
- Primary dependencies are listed in requirements.txt
- Development dependencies are specified in setup.py under extras_require['dev']
- Package is installable via pip: `pip install -e .`

### Versioning
- We follow Semantic Versioning (SemVer)
- Version is specified in setup.py

## Documentation
- We use Google-style docstrings
- Documentation is generated using Sphinx
- Documentation is hosted on GitHub Pages
- Documentation is built and deployed automatically via GitHub Actions



