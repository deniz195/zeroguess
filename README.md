# ZeroGuess: Machine Learning for Curve Fitting Parameter Estimation

[![Build Status](https://github.com/deniz195/zeroguess/actions/workflows/test.yml/badge.svg)](https://github.com/deniz195/zeroguess/actions/workflows/test.yml)
[![Coverage Status](https://codecov.io/gh/deniz195/zeroguess/branch/main/graph/badge.svg)](https://codecov.io/gh/deniz195/zeroguess)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/zeroguess.svg)](https://pypi.org/project/zeroguess/)

ZeroGuess is a Python library that simplifies the estimation of starting parameters for curve fitting by leveraging machine learning. It supports SciPy and lmfit, two widely used curve fitting libraries in the scientific Python ecosystem.

## Problem Statement

While curve fitting is a well-understood problem, the process of estimating starting parameters is not. It is a very tedious and error-prone process that often requires domain expertise, trial and error, or both. Poor initial parameter estimates can cause fitting algorithms to:
- Converge to suboptimal local minima
- Require more iterations to converge
- Fail to converge entirely

ZeroGuess uses machine learning to learn from the fitting function itself, providing optimal starting parameters without manual tuning.

## Installation

```bash
pip install zeroguess
```

## Quick Start

### Basic Usage

```python
import zeroguess
import numpy as np
from scipy import optimize

# Define function to fit
def gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-(x - center)**2 / (2 * width**2))

# Define sampling points for training
x_sampling = np.linspace(-10, 10, 100)

# Create and train parameter estimator
estimator = zeroguess.create_estimator(
    function=gaussian,
    param_ranges={
        'amplitude': (0, 10),
        'center': (-5, 5),
        'width': (0.1, 2)
    },
    independent_vars_sampling={
        'x': x_sampling
    }
)
estimator.train()

# Get parameter estimates for new data
x_data = np.linspace(-10, 10, 100)
y_data = ... # Your experimental data
initial_params = estimator.predict(x_data, y_data)

# Use in standard curve fitting
optimal_params, _ = optimize.curve_fit(
    gaussian, x_data, y_data,
    p0=initial_params
)
```

### SciPy Integration

```python
from zeroguess.integration import scipy_integration
import numpy as np

# Enhanced curve_fit with automatic parameter estimation
optimal_params, pcov = scipy_integration.curve_fit(
    gaussian, x_data, y_data,
    param_ranges={
        'amplitude': (0, 10),
        'center': (-5, 5),
        'width': (0.1, 2)
    },
    independent_vars_sampling={
        'x': np.linspace(-10, 10, 100)
    }
)
```

### lmfit Integration

```python
from zeroguess.integration import lmfit_integration
import lmfit
import numpy as np

# Enhanced lmfit Model with parameter estimation
model = lmfit_integration.Model(
    gaussian,
    param_ranges={
        'amplitude': (0, 10),
        'center': (-5, 5),
        'width': (0.1, 2)
    },
    independent_vars_sampling={
        'x': np.linspace(-10, 10, 100)
    }
)

# Standard lmfit workflow
result = model.fit(y_data, x=x_data)
```

## Features

- Automatic estimation of starting parameters for curve fitting
- Support for both SciPy and lmfit curve fitting libraries
- Neural network-based parameter estimation
- Model persistence for reuse without retraining
- Detailed diagnostics and visualization tools

## Requirements

- Python 3.10+
- Dependencies: numpy, scipy, torch, lmfit (optional)

## License

MIT
