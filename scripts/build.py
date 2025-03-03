#!/usr/bin/env python
"""
Build script for ZeroGuess.

This script automates the build process and can optionally upload to Test PyPI.
It performs the following steps:
1. Clean the build environment
2. Build both wheel and source distributions
3. Validate the built packages
4. Optionally upload to Test PyPI and test the installation

Usage:
    python scripts/build.py [--upload-test] [--test-install]

Options:
    --upload-test    Upload the built packages to Test PyPI
    --test-install   Test installation from the built wheel
"""

import argparse
import importlib.util
import os
import re
import shutil
import subprocess
import sys
import tempfile
import venv
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def run_command(cmd, cwd=None, env=None, exit_on_error=True):
    """Run a shell command and return the output."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0 and exit_on_error:
        print(f"Command failed with exit code {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        sys.exit(result.returncode)

    return result


def clean_build(deep=False):
    """Clean the build environment by removing previous builds.

    Args:
        deep: If True, performs a deeper clean including cache files and __pycache__ directories
    """
    print("\n=== Cleaning build environment ===")

    # Standard build artifacts
    paths_to_remove = ["build", "dist", "*.egg-info"]

    # Additional artifacts for deep cleaning
    if deep:
        paths_to_remove.extend(
            [
                "**/__pycache__",
                "**/*.pyc",
                "**/*.pyo",
                "**/*.pyd",
                ".pytest_cache",
                ".coverage",
                "coverage.xml",
                "htmlcov",
                ".tox",
                ".hypothesis",
            ]
        )

    project_root = Path(__file__).resolve().parent.parent

    for path_str in paths_to_remove:
        for path in list(project_root.glob(path_str)):
            if path.is_dir():
                print(f"Removing directory: {path}")
                shutil.rmtree(path)
            else:
                print(f"Removing file: {path}")
                path.unlink()

    # If deep cleaning, also remove all .so/.dll files in the package directory
    if deep:
        package_dir = project_root / "zeroguess"
        if package_dir.exists():
            for ext in [".so", ".dll", ".pyd"]:
                for binary_file in package_dir.glob(f"**/*{ext}"):
                    print(f"Removing binary file: {binary_file}")
                    binary_file.unlink()

    print("Build environment cleaned.")


def build_package():
    """Build both wheel and source distribution."""
    print("\n=== Building package ===")
    project_root = Path(__file__).resolve().parent.parent

    # Ensure build tools are installed
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip", "build", "twine"])

    # Build the package
    run_command([sys.executable, "-m", "build"], cwd=str(project_root))

    # Check if build succeeded
    dist_dir = project_root / "dist"
    if not dist_dir.exists() or not list(dist_dir.glob("*.whl")) or not list(dist_dir.glob("*.tar.gz")):
        print("Error: Build failed to produce distribution files")
        sys.exit(1)

    print("Build successful!")
    print(f"Distribution files created in {dist_dir}")
    for dist_file in dist_dir.glob("*"):
        print(f"  - {dist_file.name}")


def validate_package():
    """Validate the built packages with twine check."""
    print("\n=== Validating package ===")
    project_root = Path(__file__).resolve().parent.parent
    dist_dir = project_root / "dist"

    run_command([sys.executable, "-m", "twine", "check", str(dist_dir / "*")])

    print("Package validation passed!")


def test_installation():
    """Test installation from the built wheel in a temporary virtual environment."""
    print("\n=== Testing installation from wheel ===")
    project_root = Path(__file__).resolve().parent.parent
    dist_dir = project_root / "dist"

    wheel_files = list(dist_dir.glob("*.whl"))
    if not wheel_files:
        print("Error: No wheel file found")
        sys.exit(1)

    wheel_path = wheel_files[0]

    # Create a temporary directory for the virtual environment
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_venv = os.path.join(temp_dir, "venv")

        # Create a virtual environment
        print(f"Creating virtual environment in {temp_venv}")
        venv.create(temp_venv, with_pip=True)

        # Path to the Python executable in the virtual environment
        if sys.platform == "win32":
            venv_python = os.path.join(temp_venv, "Scripts", "python.exe")
        else:
            venv_python = os.path.join(temp_venv, "bin", "python")

        # Install the wheel
        print(f"Installing {wheel_path}")
        run_command([venv_python, "-m", "pip", "install", "--force-reinstall", str(wheel_path)])

        # Import the package and print its version
        print("Verifying installation")
        result = run_command(
            [venv_python, "-c", "import zeroguess; print(f'ZeroGuess version: {zeroguess.__version__}')"]
        )

        print(f"Installation test result: {result.stdout.strip()}")


def upload_to_test_pypi():
    """Upload the built packages to Test PyPI."""
    print("\n=== Uploading to Test PyPI ===")
    project_root = Path(__file__).resolve().parent.parent
    dist_dir = project_root / "dist"

    # Use environment variable from .env
    token = os.environ.get("TESTPYPI_TOKEN")
    if not token:
        print("ERROR: TESTPYPI_TOKEN environment variable not found")
        print("Please set it in a .env file or your environment")
        sys.exit(1)

    run_command(
        [
            sys.executable,
            "-m",
            "twine",
            "upload",
            "--repository-url",
            "https://test.pypi.org/legacy/",
            "--username",
            "__token__",
            "--password",
            token,
            str(dist_dir / "*"),
        ]
    )

    print("Upload to Test PyPI successful!")
    print("To install from Test PyPI:")
    print("pip install --index-url https://test.pypi.org/simple/ zeroguess")


def get_pyproject_version():
    """Get the version from pyproject.toml."""
    project_root = Path(__file__).resolve().parent.parent
    pyproject_path = project_root / "pyproject.toml"

    # Simple regex parsing to extract version
    with open(pyproject_path, "r") as f:
        content = f.read()

    # Look for version in [project] section
    match = re.search(r'\[project\][^\[]*version\s*=\s*"([^"]+)"', content, re.DOTALL)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")

    return match.group(1)


def get_package_version():
    """Get the version from the package __init__.py file."""
    project_root = Path(__file__).resolve().parent.parent
    init_path = project_root / "zeroguess" / "__init__.py"

    # Load the module without importing it
    spec = importlib.util.spec_from_file_location("zeroguess", init_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "__version__"):
        raise ValueError("__version__ not found in zeroguess/__init__.py")

    return module.__version__


def check_version_consistency():
    """Check that pyproject.toml and package versions match."""
    print("\n=== Checking version consistency ===")

    try:
        pyproject_version = get_pyproject_version()
        package_version = get_package_version()

        print(f"pyproject.toml version: {pyproject_version}")
        print(f"Package __version__: {package_version}")

        if pyproject_version != package_version:
            print(
                f"ERROR: Version mismatch! pyproject.toml has version {pyproject_version} but "
                f"package has version {package_version}"
            )
            print("Please update the versions to match before building.")
            sys.exit(1)

        print("Version consistency check passed!")
        return True
    except Exception as e:
        print(f"Error checking versions: {e}")
        sys.exit(1)


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Build script for ZeroGuess")
    parser.add_argument("--upload-test", action="store_true", help="Upload the built packages to Test PyPI")
    parser.add_argument("--test-install", action="store_true", help="Test installation from the built wheel")
    parser.add_argument("--deep-clean", action="store_true", help="Perform a deeper clean including cache files")
    parser.add_argument(
        "--skip-version-check",
        action="store_true",
        help="Skip version consistency check between pyproject.toml and __init__.py",
    )

    args = parser.parse_args()

    # Check version consistency before proceeding
    if not args.skip_version_check:
        check_version_consistency()

    clean_build(deep=args.deep_clean)
    build_package()
    validate_package()

    if args.test_install:
        test_installation()

    if args.upload_test:
        upload_to_test_pypi()


if __name__ == "__main__":
    main()
