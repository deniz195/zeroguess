from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="zeroguess",
    version="0.1.0",
    author="ZeroGuess Team",
    author_email="info@zeroguess.org",
    description="Machine Learning for Curve Fitting Parameter Estimation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zeroguess/zeroguess",
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
        "dev": ["pytest", "pytest-cov", "black", "flake8", "mypy"],
    },
)
