#!/usr/bin/env python
"""
Script to run security checks on the codebase.

Usage:
    python scripts/security_check.py
"""
import subprocess
import sys
from pathlib import Path

# Define project directories
PROJECT_ROOT = Path(__file__).parent.parent
PYTHON_DIRS = ["zeroguess", "tests", "examples"]
PYTHON_DIRS_STR = " ".join(PYTHON_DIRS)


def run_command(command, description):
    """Run a shell command and print its output."""
    print(f"\n\033[1;34m{description}\033[0m")
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode != 0:
        print(f"\033[1;31mFailed: {description}\033[0m")
        return False
    print(f"\033[1;32mSuccess: {description}\033[0m")
    return True


def check_bandit():
    """Run bandit security checks."""
    return run_command(f"bandit -r {PYTHON_DIRS_STR} -ll", "Running bandit security checks")


def check_safety():
    """Check dependencies for known security vulnerabilities."""
    return run_command("safety check", "Checking dependencies for known security vulnerabilities")


def main():
    """Main entry point."""
    print("\033[1;34mRunning security checks...\033[0m")

    # Check if bandit is installed
    try:
        subprocess.run(["bandit", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\033[1;31mError: bandit is not installed.\033[0m")
        print("Please install it with: pip install bandit")
        return 1

    # Check if safety is installed
    try:
        subprocess.run(["safety", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\033[1;31mError: safety is not installed.\033[0m")
        print("Please install it with: pip install safety")
        return 1

    success = True

    # Run bandit
    success = check_bandit() and success

    # Run safety
    success = check_safety() and success

    if success:
        print("\n\033[1;32mAll security checks passed!\033[0m")
    else:
        print("\n\033[1;31mSome security checks failed. Please fix the issues and try again.\033[0m")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
