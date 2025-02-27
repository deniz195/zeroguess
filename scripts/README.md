# Development Scripts

This directory contains utility scripts for development.

## Code Quality Tools

### check_all.py

A script to run all code quality checks at once.

Usage:
```bash
# Run all code quality checks
python scripts/check_all.py
```

You can also make the script executable and run it directly:
```bash
chmod +x scripts/check_all.py
./scripts/check_all.py
```

### quality.py

A script to run code quality tools manually.

Usage:
```bash
# Run all code quality tools (format and lint)
python scripts/quality.py

# Run only formatters (black, isort, autoflake, autopep8)
python scripts/quality.py format

# Run only linters (flake8, mypy, vulture)
python scripts/quality.py lint

# Check formatting and linting without making changes
python scripts/quality.py check

# Run pre-commit hooks on all files
python scripts/quality.py pre-commit

# Automatically fix common flake8 issues
python scripts/quality.py fix
```

You can also make the script executable and run it directly:
```bash
chmod +x scripts/quality.py
./scripts/quality.py
```

### setup_hooks.py

A script to install pre-commit hooks.

Usage:
```bash
# Install pre-commit hooks
python scripts/setup_hooks.py
```

You can also make the script executable and run it directly:
```bash
chmod +x scripts/setup_hooks.py
./scripts/setup_hooks.py
```

### check_deps.py

A script to check for outdated dependencies.

Usage:
```bash
# Check for outdated dependencies
python scripts/check_deps.py
```

You can also make the script executable and run it directly:
```bash
chmod +x scripts/check_deps.py
./scripts/check_deps.py
```

### security_check.py

A script to run security checks on the codebase.

Usage:
```bash
# Run security checks
python scripts/security_check.py
```

You can also make the script executable and run it directly:
```bash
chmod +x scripts/security_check.py
./scripts/security_check.py
``` 