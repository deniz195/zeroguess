#!/usr/bin/env python
"""
Script to check for outdated dependencies.

Usage:
    python scripts/check_deps.py
"""
import subprocess
import sys
from pathlib import Path


def main():
    """Check for outdated dependencies."""
    print("\033[1;34mChecking for outdated dependencies...\033[0m")
    
    # Check if pip-tools is installed
    try:
        subprocess.run(["pip-compile", "--version"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("\033[1;33mWarning: pip-tools is not installed.\033[0m")
        print("For better dependency management, install it with: pip install pip-tools")
    
    # Check for outdated packages
    result = subprocess.run(
        ["pip", "list", "--outdated", "--format=columns"],
        text=True,
        capture_output=True
    )
    
    if result.returncode != 0:
        print("\033[1;31mFailed to check for outdated packages.\033[0m")
        return 1
    
    if not result.stdout.strip():
        print("\033[1;32mAll dependencies are up to date!\033[0m")
        return 0
    
    print("\n\033[1;33mOutdated dependencies:\033[0m")
    print(result.stdout)
    
    print("\nTo update a package, run: pip install --upgrade <package>")
    print("To update all packages (not recommended), run: pip list --outdated --format=freeze | grep -v '\\\\-e' | cut -d = -f 1 | xargs -n1 pip install -U")
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 