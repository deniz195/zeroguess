# ZeroGuess Functions Submodule

This submodule provides a collection of standard fitting functions and utilities for use with the ZeroGuess library. The functions are designed to be used as target functions for parameter estimation tasks.

## Available Functions

The submodule includes the following standard functions:

- **GaussianFunction**: A standard Gaussian/normal distribution function.
- **MultiPeakGaussianFunction**: A multi-peak Gaussian function that combines multiple Gaussian peaks.
- **DampedSineFunction**: A damped sine wave function.
- **LinearFunction**: A simple linear function.

## Usage

### Basic Usage

```python
from zeroguess.functions import GaussianFunction

# Create a Gaussian function
gaussian = GaussianFunction()

# Get information about the function
print(gaussian.name)  # "Gaussian Function"
print(gaussian.param_ranges)  # Dictionary of parameter ranges
print(gaussian.param_descriptions)  # Dictionary of parameter descriptions

# Evaluate the function with specific parameters
import numpy as np
x = np.linspace(-5, 5, 100)
params = {'amplitude': 1.0, 'center': 0.0, 'width': 1.0}
y = gaussian(x, **params)
```

### Generating Data

Each function provides a `generate_data` method that returns both the independent variables and the corresponding function values:

```python
from zeroguess.functions import DampedSineFunction

# Create a damped sine function
damped_sine = DampedSineFunction()

# Generate data with specific parameters
params = {
    'amplitude': 3.0,
    'frequency': 0.5,
    'phase': 0.0,
    'decay': 0.2
}
indep_vars, y_data = damped_sine.generate_data(params)

# The independent variables are returned as a dictionary
x = indep_vars['x']
```

### Random Parameters

You can generate random parameters within the valid ranges:

```python
from zeroguess.functions import MultiPeakGaussianFunction

# Create a multi-peak Gaussian function with 3 peaks
multi_gaussian = MultiPeakGaussianFunction(n_peaks=3)

# Generate random parameters
random_params = multi_gaussian.get_random_params()
print(random_params)
```

### Adding Noise

The submodule includes utility functions for adding noise to your data:

```python
from zeroguess.functions import GaussianFunction, add_gaussian_noise

# Create a Gaussian function and generate data
gaussian = GaussianFunction()
indep_vars, y_data = gaussian.generate_data({'amplitude': 2.0, 'center': 0.0, 'width': 1.5})

# Add Gaussian noise with a standard deviation of 0.1
y_noisy = add_gaussian_noise(y_data, sigma=0.1)
```

## Extending with Custom Functions

You can create your own custom fitting functions by subclassing the `FittingFunction` base class:

```python
from zeroguess.functions import FittingFunction
import numpy as np

class ExponentialFunction(FittingFunction):
    @property
    def name(self):
        return "Exponential Function"
    
    @property
    def param_ranges(self):
        return {
            'amplitude': (0.1, 10.0),
            'decay': (0.01, 2.0),
            'offset': (-5.0, 5.0)
        }
    
    @property
    def param_descriptions(self):
        return {
            'amplitude': "Amplitude of the exponential function",
            'decay': "Decay rate of the exponential function",
            'offset': "Vertical offset of the function"
        }
    
    @property
    def default_independent_vars(self):
        return {'x': np.linspace(0, 10, 100)}
    
    def __call__(self, x, amplitude, decay, offset):
        return amplitude * np.exp(-decay * x) + offset
```

## Example Script

See the `examples/functions_example.py` script for a complete demonstration of using the functions submodule. 