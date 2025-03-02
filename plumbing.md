# Plumbing

## Overview
This file describes the ZeroGuess project's coding and linting standards, CI/CD pipeline, Python package details, and development workflow.

## Code Quality Standards

### Formatting
- **Black**: We use Black for code formatting with a line length of 120 characters.
- **isort**: We use isort for sorting imports, configured to be compatible with Black.

### Linting
- **Flake8**: We use Flake8 for linting with the following plugins:
  - flake8-bugbear: For detecting potential bugs
  - flake8-docstrings: For docstring style checking (disabled)
  - flake8-import-order: For import order checking (disabled)
  - pep8-naming: For PEP 8 naming convention checking (disabled)

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

### Scripts

We have a few scripts to help with development.

- `scripts/quality.py` - Runs linters, formatters, and tests.


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
black --line-length=120 zeroguess tests

# Run isort
isort zeroguess tests
```

### Auto-fixing Flake8 Issues
While Flake8 itself doesn't have auto-fix capabilities, we use several tools to automatically fix common issues:

```bash
# Fix unused imports with autoflake
find zeroguess tests examples -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables

# Fix import order with isort
isort zeroguess tests examples

# Fix code style issues with black
black --line-length=120 zeroguess tests examples

# Fix PEP 8 style issues with autopep8
find zeroguess tests examples -name '*.py' | xargs autopep8 --in-place --aggressive --max-line-length=120
```

These tools can be installed with:
```bash
pip install autoflake autopep8
```

For a comprehensive fix, run them in this order:
1. autoflake (removes unused imports)
2. isort (sorts imports)
3. autopep8 (fixes PEP 8 issues)
4. black (formats code according to our style)

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



