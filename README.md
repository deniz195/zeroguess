# ZeroGuess: Machine Learning for Curve Fitting Parameter Estimation

[![Build Status](https://github.com/deniz195/zeroguess/actions/workflows/test.yml/badge.svg)](https://github.com/deniz195/zeroguess/actions/workflows/test.yml)
[![Coverage Status](https://codecov.io/gh/deniz195/zeroguess/branch/main/graph/badge.svg)](https://codecov.io/gh/deniz195/zeroguess)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/zeroguess.svg)](https://pypi.org/project/zeroguess/)
[![Benchmark Results](https://img.shields.io/badge/benchmarks-view%20results-blue)](https://deniz195.github.io/zeroguess/)

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
import numpy as np
from zeroguess.functions import WaveletFunction, add_gaussian_noise
from zeroguess.integration import ZeroGuessModel

# Create a double peakGaussian function
wavelet = WaveletFunction()

# Create some experimental data
true_params = wavelet.get_random_params()
x_data = np.linspace(-10, 10, 100)
y_data = add_gaussian_noise(wavelet(x_data, **true_params), sigma=0.1)
```

```python
import zeroguess
from scipy import optimize

# Create and train parameter estimator
estimator = zeroguess.create_estimator(
    function=wavelet,
    param_ranges={
        'frequency': (0, 10),
        'phase': (-5, 5),
        'position': (0.1, 2),
        'width': (0, 10),
    },
    independent_vars_sampling={
        'x': x_data
    }
)
estimator.train()

# Get parameter estimates for experimental data
initial_params = estimator.predict(x_data, y_data)

# Use in standard curve fitting
optimal_params, _ = optimize.curve_fit(
    wavelet, x_data, y_data,
    p0=initial_params
)
```

### lmfit Integration

```python

# Enhanced lmfit Model with parameter estimation
model = ZeroGuessModel(
    wavelet,
    independent_vars_sampling={"x": x_data},
    estimator_settings={
        "make_canonical": wavelet.get_canonical_params,
        # Configure training parameters
        # "n_samples": 1000,
        # "n_epochs": 200,
        # "validation_split": 0.2,
        # "add_noise": True,
        # "noise_level": 0.1,
        # 'verbose': True
        "snapshot_path": "model_dg.pth", # saves and loads model automatically
    },
)

model.set_param_hint("frequency", min=0, max=10)
model.set_param_hint("phase", min=-5, max=5)
model.set_param_hint("position", min=0.1, max=2)
model.set_param_hint("width", min=0, max=10)

# Standard lmfit workflow
params = model.guess(y_data, x=x_data)

result = model.fit(y_data, x=x_data, params=params)
```


### SciPy Integration

```python
from zeroguess.integration import scipy_integration
import numpy as np

# Enhanced curve_fit with automatic parameter estimation
optimal_params, pcov = scipy_integration.curve_fit(
    wavelet, x_data, y_data,
    param_ranges={
        'frequency': (0, 10),
        'phase': (-5, 5),
        'position': (0.1, 2),
        'width': (0, 10),
    },
    independent_vars_sampling={
        'x': np.linspace(-10, 10, 100)
    }
)
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

## Performance Benchmarks

ZeroGuess is benchmarked regularly to ensure optimal performance. View the latest [benchmark results](https://deniz195.github.io/zeroguess/).
