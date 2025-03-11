# ZeroGuess: Machine Learning for Curve Fitting Parameter Estimation

[![Build Status](https://github.com/deniz195/zeroguess/actions/workflows/test.yml/badge.svg)](https://github.com/deniz195/zeroguess/actions/workflows/test.yml)
[![Coverage Status](https://codecov.io/gh/deniz195/zeroguess/branch/main/graph/badge.svg)](https://codecov.io/gh/deniz195/zeroguess)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Versions](https://img.shields.io/pypi/pyversions/zeroguess.svg)](https://pypi.org/project/zeroguess/)
[![Benchmark Results](https://img.shields.io/badge/benchmarks-view%20results-blue)](https://deniz195.github.io/zeroguess/)

ZeroGuess uses machine learning to improve curve fitting parameter estimation. It generates optimal starting parameters for fitting functions, reducing computation time and increasing fitting reliability.

## Problem Statement

Curve fitting in scientific applications often encounters issues with initial parameter estimation:
- Convergence to local rather than global minima
- High computational cost for complex functions
- Failed convergence with inadequate starting points

Conventional approaches rely on domain expertise, manual adjustment, or computationally intensive global optimization methods.

ZeroGuess addresses these limitations through machine learning models trained on synthetic data, providing effective parameter estimates that allow simpler fitting algorithms to achieve results comparable to advanced methods with significantly reduced computation time.

## Installation

```bash
pip install zeroguess
```

## Quick Start

### Setup experimental data

```python
import numpy as np
from zeroguess.functions import WaveletFunction, add_gaussian_noise

# Create a double peakGaussian function
wavelet = WaveletFunction()

# Create some experimental data
true_params = wavelet.get_random_params()
x_data = np.linspace(-10, 10, 100)
y_data = add_gaussian_noise(wavelet(x_data, **true_params), sigma=0.1)
```

### lmfit Integration

```python
from zeroguess.integration import ZeroGuessModel

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

model.set_param_hint("frequency", min=0.05, max=1.0)
model.set_param_hint("phase", min=0.0, max=2.0 * np.pi)
model.set_param_hint("position", min=5.0, max=15.0)
model.set_param_hint("width", min=0.1, max=3.0)

# Standard lmfit workflow
params = model.guess(y_data, x=x_data)

result = model.fit(y_data, x=x_data, params=params)
```

### Scipy usage

```python
import zeroguess
from scipy import optimize

# Create and train parameter estimator
estimator = zeroguess.create_estimator(
    function=wavelet,
    param_ranges={
        "frequency": (0.05, 1.0),
        "phase": (0.0, 2.0 * np.pi),
        "position": (5.0, 15.0),
        "width": (0.1, 3.0),
    },
    independent_vars_sampling={
        'x': x_data
    },
    snapshot_path="model_dg_plain.pth", # saves and loads model automatically
)

if not estimator.is_trained:
    estimator.train()

# Get parameter estimates for experimental data
initial_params = estimator.predict(x_data, y_data)

# Use in standard curve fitting
optimal_params, _ = optimize.curve_fit(
    wavelet, x_data, y_data,
    p0=list(initial_params.values())
)
```

## Background



### Canonical Representations

Many fitting functions produce identical outputs with different parameter combinations. For example, in a double Gaussian function, swapping the parameters of the two peaks produces the same curve:

```python
# These two parameter sets produce identical curves
params1 = {"amplitude1": 2, "center1": -2, "width1": 1, 
           "amplitude2": 3, "center2": 2, "width2": 0.5}
params2 = {"amplitude1": 3, "center2": 2, "width2": 0.5,
           "amplitude2": 2, "center1": -2, "width1": 1}
```

You can provide ZeroGuess a function for `make_canonical` to transform parameters into a standardized form (e.g., ordering peaks from left to right), enabling the model to learn a consistent mapping and provide more reliable parameter estimates.

### Performance and Advanced Fitting Methods

Accurate starting parameters are essential for local optimization methods like `least_squares`. Global optimization methods like `dual_annealing` find better solutions but require substantially more computation.  
[![Benchmark Results](https://img.shields.io/badge/benchmarks-view%20results-blue)](https://deniz195.github.io/zeroguess/latest/lmfit_comparison/double_gaussian/report.html )

Benchmarks indicate that ZeroGuess improves the performance of `least_squares` fits to comparable levels with `dual_annealing` (~70% success rate) while reducing computation time by a factor of 100 (from 1150ms to 12ms).

This makes ZeroGuess useful for complex fitting functions with multiple parameters and for large datasets where computation efficiency is important.

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
