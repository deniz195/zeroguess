#!/usr/bin/env python
"""
Script to set up pre-commit hooks.

Usage:
    python scripts/setup_hooks.py
"""
import subprocess
import sys


def main():
    """Install pre-commit hooks."""
    print("\033[1;34mInstalling pre-commit hooks...\033[0m")

    # Check if pre-commit is installed
    try:
        subprocess.run(["pre-commit", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\033[1;31mError: pre-commit is not installed.\033[0m")
        print("Please install it with: pip install pre-commit")
        return 1

    # Install the hooks
    result = subprocess.run(["pre-commit", "install"], text=True)
    if result.returncode != 0:
        print("\033[1;31mFailed to install pre-commit hooks.\033[0m")
        return 1

    print("\033[1;32mPre-commit hooks installed successfully!\033[0m")
    print("\nHooks will now run automatically on commit.")
    print("You can also run them manually with: pre-commit run --all-files")

    return 0


if __name__ == "__main__":
    sys.exit(main())
