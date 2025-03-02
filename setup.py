#!/usr/bin/env python
"""
Setup script for ZeroGuess package.
"""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zeroguess",
    version="0.1.0",
    author="Deniz Bozyigit",
    author_email="deniz195@gmail.com",
    description="Machine Learning for Curve Fitting Parameter Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/deniz195/zeroguess",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "torch>=1.10.0",
        "matplotlib>=3.5.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "lmfit": ["lmfit>=1.0.0"],
        "dev": [
            # Testing
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "pytest-benchmark>=4.0.0",
            "pytest-xdist>=3.0.0",
            
            # Code formatting
            "black>=23.3.0",
            "isort>=5.12.0",
            
            # Linting
            "flake8>=6.0.0",
            "flake8-bugbear>=23.3.23",
            "flake8-pyproject>=1.2.0",
            # "flake8-docstrings>=1.7.0",
            # "flake8-import-order>=0.18.2",
            # "pep8-naming>=0.13.3",
            "mypy>=1.3.0",
            "vulture>=2.7",
            
            # Security
            "bandit>=1.7.5",
            "safety>=2.3.5",
            
            # Development tools
            "pre-commit>=3.3.2",
            "pip-tools>=6.13.0",
            
            # Documentation
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
)
