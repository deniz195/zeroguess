#!/usr/bin/env python
"""
Script to run all code quality checks at once.

Usage:
    python scripts/check_all.py
"""
import subprocess
import sys
from pathlib import Path

# Import other scripts
script_dir = Path(__file__).parent
sys.path.append(str(script_dir))

import quality
import security_check
import check_deps


def main():
    """Run all code quality checks."""
    print("\033[1;34m=== Running All Code Quality Checks ===\033[0m")
    
    success = True
    
    # Run code quality checks
    print("\n\033[1;34m=== Code Quality Checks ===\033[0m")
    result = quality.main()
    success = (result == 0) and success
    
    # Run security checks
    print("\n\033[1;34m=== Security Checks ===\033[0m")
    result = security_check.main()
    success = (result == 0) and success
    
    # Check dependencies
    print("\n\033[1;34m=== Dependency Checks ===\033[0m")
    result = check_deps.main()
    success = (result == 0) and success
    
    if success:
        print("\n\033[1;32m=== All checks passed! ===\033[0m")
    else:
        print("\n\033[1;31m=== Some checks failed. Please fix the issues and try again. ===\033[0m")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main()) 