#!/usr/bin/env python
"""
Script to run code quality tools manually.

Usage:
    python scripts/quality.py [command]

Commands:
    format: Run formatters (black, isort, autoflake, autopep8)
    lint: Run linters (flake8, mypy, vulture)
    check: Run formatters and linters in check mode (no changes)
    all: Run all code quality tools
    fix: Run auto-fixers for common flake8 issues
"""
import argparse
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


def format_code(check_only=False):
    """Run code formatters."""
    success = True
    
    # Run autoflake to remove unused imports
    if not check_only:
        cmd = f"find {PYTHON_DIRS_STR} -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables"
        success = run_command(cmd, "Removing unused imports with autoflake") and success
    
    # Run isort
    check_flag = "--check-only --diff" if check_only else ""
    cmd = f"isort {check_flag} {PYTHON_DIRS_STR}"
    action = "Checking" if check_only else "Formatting"
    success = run_command(cmd, f"{action} imports with isort") and success
    
    # Run autopep8 for PEP 8 fixes
    if not check_only:
        cmd = f"find {PYTHON_DIRS_STR} -name '*.py' | xargs autopep8 --in-place --aggressive --max-line-length=120"
        success = run_command(cmd, "Fixing PEP 8 issues with autopep8") and success
    
    # Run black
    check_flag = "--check --diff" if check_only else ""
    cmd = f"black {check_flag} --line-length=120 {PYTHON_DIRS_STR}"
    success = run_command(cmd, f"{action} code with black") and success
    
    return success


def lint_code():
    """Run code linters."""
    success = True
    
    # Run flake8
    cmd = f"flake8 {PYTHON_DIRS_STR}"
    success = run_command(cmd, "Linting code with flake8") and success
    
    # # Run mypy
    # cmd = f"mypy zeroguess"
    # success = run_command(cmd, "Type checking with mypy") and success
    
    # # Run vulture
    # cmd = f"vulture {PYTHON_DIRS_STR} --min-confidence=80"
    # success = run_command(cmd, "Checking for unused code with vulture") and success
    
    return success


def fix_flake8_issues():
    """Fix common flake8 issues automatically."""
    success = True
    
    # Remove unused imports
    cmd = f"find {PYTHON_DIRS_STR} -name '*.py' | xargs autoflake --in-place --remove-all-unused-imports --remove-unused-variables"
    success = run_command(cmd, "Removing unused imports with autoflake") and success
    
    # Sort imports
    cmd = f"isort {PYTHON_DIRS_STR}"
    success = run_command(cmd, "Sorting imports with isort") and success
    
    # Fix PEP 8 issues
    cmd = f"find {PYTHON_DIRS_STR} -name '*.py' | xargs autopep8 --in-place --aggressive --max-line-length=120"
    success = run_command(cmd, "Fixing PEP 8 issues with autopep8") and success
    
    # Format code
    cmd = f"black --line-length=120 {PYTHON_DIRS_STR}"
    success = run_command(cmd, "Formatting code with black") and success
    
    return success


def run_pre_commit():
    """Run pre-commit on all files."""
    return run_command("pre-commit run --all-files", "Running pre-commit hooks on all files")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run code quality tools")
    parser.add_argument(
        "command",
        choices=["format", "lint", "check", "all", "pre-commit", "fix"],
        default="all",
        nargs="?",
        help="Command to run (format, lint, check, all, pre-commit, or fix)",
    )
    args = parser.parse_args()
    
    success = True
    
    if args.command == "format":
        success = format_code(check_only=False)
    elif args.command == "lint":
        success = lint_code()
    elif args.command == "check":
        success = format_code(check_only=True) and lint_code()
    elif args.command == "pre-commit":
        success = run_pre_commit()
    elif args.command == "fix":
        success = fix_flake8_issues()
    else:  # all
        success = format_code(check_only=False) and lint_code()
    
    if success:
        print("\n\033[1;32mAll code quality checks passed!\033[0m")
    else:
        print("\n\033[1;31mSome code quality checks failed. Please fix the issues and try again.\033[0m")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 