# ZeroGuess: Architecture Design Document

## System Overview

ZeroGuess is a Python library designed to simplify the estimation of starting parameters for curve fitting by leveraging machine learning. This document outlines the architectural design of the system, key components, and the rationale behind these design decisions.

## Design Considerations and Tradeoffs

### Simplicity vs. Flexibility
Given that the primary users are domain experts rather than ML specialists, we prioritize ease of use and simplicity in the API design. However, we must balance this with the flexibility needed to handle a variety of curve fitting scenarios.

**Tradeoff Analysis**: A highly flexible system with many configuration options would accommodate more use cases but would increase complexity for users. Conversely, an overly simplified system might not handle the diversity of curve fitting problems. Our design leans toward simplicity with sensible defaults while providing access to advanced options for expert users when needed.

### Integration vs. Independence
The library needs to integrate with existing curve fitting libraries (SciPy and lmfit) while maintaining its own identity and functionality.

**Tradeoff Analysis**: Deep integration provides a seamless experience but creates dependencies on specific versions or implementations of these libraries. A more independent design might be more robust but would require users to do more "glue" work. We'll adopt a facade pattern that provides tight integration but with clear boundaries.

### Performance vs. Maintainability
The system needs to train neural networks in a reasonable timeframe (1-3 minutes) on CPU hardware for problems of modest scale.

**Tradeoff Analysis**: Highly optimized training code would improve performance but reduce maintainability and readability. Given the relatively small scale of the problems (≤10 parameters, ≤5 independent variables, ≤1000 data points), we prioritize code clarity and maintainability over extreme performance optimizations, while ensuring that performance goals are still met.

### General vs. Specialized Architectures
Neural network architectures can be general or specialized for specific types of curve fitting problems.

**Tradeoff Analysis**: General architectures work across more problems but may underperform on specific ones. Specialized architectures can be more effective but limit applicability. We'll implement a modular approach with a general architecture by default and the ability to extend with specialized architectures for common fitting problems.

### Runtime vs. Development Complexity
Hiding ML complexity from users shifts complexity from runtime (user experience) to development (implementation).

**Tradeoff Analysis**: Automation of ML aspects simplifies usage but requires sophisticated implementation to handle the variety of fitting functions and parameter spaces. This is an acceptable tradeoff since development effort is a one-time cost while user experience is ongoing.

## Core Architecture Components

### 1. Core Library Module Structure

```
zeroguess/
├── __init__.py
├── estimators/          # Parameter estimation models
│   ├── __init__.py
│   ├── base.py          # Base estimator interfaces
│   ├── nn_estimator.py  # Neural network implementation
│   └── factory.py       # Factory for creating estimators
├── data/                # Data handling components
│   ├── __init__.py
│   ├── generators.py    # Synthetic data generation
│   ├── preprocessors.py # Data preprocessing utilities
│   └── validators.py    # Input validation utilities
├── integration/         # Integration with fitting libraries
│   ├── __init__.py
│   ├── scipy_adapter.py # SciPy integration
│   └── lmfit_adapter.py # lmfit integration
├── training/            # Model training components
│   ├── __init__.py
│   ├── trainer.py       # Training loop implementation
│   └── metrics.py       # Training metrics and evaluation
├── io/                  # Persistence components
│   ├── __init__.py
│   ├── serialization.py # Model serialization/deserialization
│   └── storage.py       # Storage utilities
└── utils/               # Utility functions
    ├── __init__.py
    ├── logging.py       # Logging utilities
    └── visualization.py # Visualization tools
```

### 2. Component Description

#### Parameter Estimator Interface
- **Purpose**: Defines the interface for parameter estimation models
- **Responsibilities**: 
  - Define API for training and prediction
  - Provide base functionality common to all estimators
  - Implement model selection logic

#### Neural Network Estimator
- **Purpose**: Implements parameter estimation using neural networks
- **Responsibilities**:
  - Define network architecture
  - Implement forward/backward pass
  - Handle tensor operations

#### Data Generation
- **Purpose**: Generate synthetic data for training
- **Responsibilities**:
  - Sample parameters from specified ranges
  - Compute function values for given parameters
  - Add configurable noise to training data

#### Library Integration
- **Purpose**: Integrate with SciPy and lmfit
- **Responsibilities**:
  - Adapt ZeroGuess functionality to library-specific APIs
  - Extract function information from library objects
  - Inject estimated parameters back into library workflow

#### Training Subsystem
- **Purpose**: Train parameter estimation models
- **Responsibilities**:
  - Implement training loops
  - Manage optimization process
  - Track and report training metrics

#### Persistence Module
- **Purpose**: Save and load trained models
- **Responsibilities**:
  - Serialize models to/from disk
  - Version model files
  - Validate model compatibility during loading

#### Utilities
- **Purpose**: Provide common functionality across modules
- **Responsibilities**:
  - Logging and error reporting
  - Visualization of results
  - Validation helpers

## Data Flow

1. **Configuration Phase**:
   - User provides fitting function and parameter ranges
   - System validates inputs and configures data generation

2. **Data Generation Phase**:
   - System generates random parameter sets within ranges
   - System evaluates fitting function with these parameters
   - Training dataset is created from parameter-output pairs

3. **Training Phase**:
   - Neural network is initialized with appropriate architecture
   - Model is trained on synthetic data
   - Training metrics are monitored for convergence/issues

4. **Inference Phase**:
   - User provides experimental data
   - System preprocesses the data
   - Model predicts initial parameters
   - System validates prediction quality
   - Parameters are returned to user or injected into fitting workflow

5. **Persistence (Optional)**:
   - Trained model is serialized to disk
   - Model can be loaded for later use without retraining

## Key Interfaces

### Primary User API

```python
# Simple usage example
import zeroguess
import numpy as np
from scipy import optimize

# Define function to fit
def gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-(x - center)**2 / (2 * width**2))

# Define independent variable sampling points
x_sampling = np.linspace(-10, 10, 100)  # Domain for training

# Create parameter estimator
estimator = zeroguess.create_estimator(
    function=gaussian,
    param_ranges={
        'amplitude': (0, 10),
        'center': (-5, 5),
        'width': (0.1, 2)
    },
    independent_vars_sampling={
        'x': x_sampling  # Sampling points for the independent variable
    }
)

# Train the estimator
estimator.train()

# Use with experimental data
x_data = np.linspace(-10, 10, 100)
y_data = noisy_gaussian_data(x_data)  # Some noisy data

# Get parameter estimates
initial_params = estimator.predict(x_data, y_data)

# Use in standard curve fitting
optimal_params, _ = optimize.curve_fit(
    gaussian, x_data, y_data,
    p0=initial_params  # Use our estimated parameters
)
```

### SciPy Integration

```python
# Enhanced scipy.optimize.curve_fit with automatic initial parameter estimation
from zeroguess.integration import scipy_integration
import numpy as np

# Define independent variable sampling points for training
x_sampling = np.linspace(-10, 10, 100)

# Standard curve_fit signature with automatic initial parameter estimation
optimal_params, pcov = scipy_integration.curve_fit(
    gaussian, x_data, y_data,
    param_ranges={  # Additional parameter for estimation
        'amplitude': (0, 10),
        'center': (-5, 5),
        'width': (0.1, 2)
    },
    independent_vars_sampling={
        'x': x_sampling  # Sampling points for training
    }
)
```

### lmfit Integration

```python
# Enhanced lmfit with automatic initial parameter estimation
from zeroguess.integration import lmfit_integration
import lmfit
import numpy as np

# Define independent variable sampling points for training
x_sampling = np.linspace(-10, 10, 100)

# Create an lmfit model with automatic parameter estimation
model = lmfit_integration.Model(
    gaussian,
    param_ranges={  # Additional parameter for estimation
        'amplitude': (0, 10),
        'center': (-5, 5),
        'width': (0.1, 2)
    },
    independent_vars_sampling={
        'x': x_sampling  # Sampling points for training
    }
)

# Standard lmfit workflow continues
result = model.fit(y_data, x=x_data)
```

### Enhanced lmfit Integration (Utilizing lmfit's Parameter Bounds)

```python
# Enhanced lmfit with automatic initial parameter estimation using lmfit's parameter bounds
from zeroguess.integration import lmfit_integration
import lmfit
import numpy as np

# Define independent variable sampling points for training
x_sampling = np.linspace(-10, 10, 100)

# Create an lmfit model with automatic parameter estimation
model = lmfit_integration.Model(
    gaussian,
    independent_vars_sampling={
        'x': x_sampling  # Sampling points for training
    },
    auto_extract_bounds=True  # Extract bounds from params automatically
)

# Set parameter bounds (will be automatically used for parameter estimation)
params = model.make_params()
params['amplitude'].min = 0
params['amplitude'].max = 10
params['center'].min = -5
params['center'].max = 5
params['width'].min = 0.1
params['width'].max = 2

# Standard lmfit workflow continues
result = model.fit(y_data, params=params, x=x_data)
```

## lmfit Integration Enhancement

To better align with lmfit's design patterns, we should enhance our `lmfit_adapter.py` to complement the built-in `guess()` method rather than working around it. The improved approach will:

1. **Focus on the gap in lmfit's functionality**: 
   - ZeroGuess is most valuable when a model doesn't implement the `guess()` method
   - Our implementation should provide a robust alternative when native parameter estimation is unavailable
   - Rather than duplicating parameter bounds, leverage lmfit's parameter constraint system

2. **Simplified integration approach**:
   - Make ZeroGuess-enhanced models use their own parameter estimation when `guess()` is not implemented
   - Properly handle the case where parameters are provided vs. when they need to be estimated
   - Ensure our implementation follows lmfit's parameter handling patterns

3. **Integration with lmfit workflow**:
   - Honor lmfit's parameter constraints when using ZeroGuess
   - Preserve user-specified parameter values rather than overriding them
   - Allow for proper serialization and model saving/loading

This approach focuses on filling the functionality gap in lmfit models that don't provide their own parameter estimation, without attempting to replace or enhance existing `guess()` implementations.

## Neural Network Architecture

The default neural network architecture will follow a modular design:

1. **Input Layer**: 
   - Accepts the data points (x, y values) with variable shapes to accommodate different numbers of independent variables and data points.

2. **Feature Extraction Layers**:
   - Convolutional or attention-based layers to extract patterns from the data.
   - Dimensionality reduction to create fixed-size representations.

3. **Parameter Prediction Layers**:
   - Fully connected layers that map from the extracted features to parameter estimates.
   - Output layer normalized to the parameter ranges.

4. **Loss Function**:
   - Primary: Mean squared error between predicted output curve and target curve.
   - Regularization: Penalties to encourage realistic parameter values.

The specific number of layers and neurons will be determined during implementation, with sensible defaults chosen to balance complexity and performance for the expected scale (≤10 parameters, ≤5 independent variables, ≤1000 data points).

## Extension Points

The architecture includes several well-defined extension points:

1. **Custom Estimators**: Users can implement custom parameter estimator classes.
2. **Custom Neural Architectures**: Expert users can define specialized architectures.
3. **Additional Library Integrations**: New adapters can be added for other curve fitting libraries.
4. **Custom Data Generators**: Special sampling strategies can be implemented.
5. **Enhanced Visualizations**: Additional visualization tools can be added.

## Deployment Considerations

Since the library will run locally on a user's machine, deployment is simplified:

1. **Dependencies**: 
   - Python 3.10+
   - PyTorch as the primary ML framework
   - SciPy and/or lmfit as fitting libraries
   - Minimal additional dependencies

2. **Hardware Requirements**:
   - All training on single-core CPU
   - Potential future extension for GPU acceleration
   - Memory requirements modest given problem scale

3. **Installation**:
   - Standard Python package installation via pip
   - Optional extras for specific integrations

4. **Example Usage Guidelines**:
   - Examples should always be executed from the examples directory as the working directory
   - Example: `cd examples && python gaussian_example.py --method all`
   - This ensures proper relative imports and file path handling
   - All example scripts include command-line arguments for configuration

## Future Extensibility

The modular architecture allows for future enhancements:

1. **GPU Acceleration**: The neural network components can be extended to use GPU acceleration when available.
2. **Distributed Training**: For larger problems, training could be distributed.
3. **Additional Fitting Libraries**: Support for more libraries can be added through the adapter pattern.
4. **Pre-trained Models**: A repository of pre-trained models for common functions could be developed.
5. **Web Service**: The parameter estimation could be exposed as a web service.

## Implementation Strategy

The recommended implementation approach is:

1. **Phase 1**: Core functionality
   - Basic parameter estimation interface
   - Data generation subsystem
   - Simple neural network implementation
   - SciPy integration
   - lmfit integration

1. **Phase 2**: Verification
   - Add verification examples as unit tests
   - Add verification tests for the core functionality

2. **Phase 3**: Enhanced functionality
   - Improved neural architectures
   - Model persistence
   - Comprehensive error handling

3. **Phase 4**: Optimization and extensions
   - Performance optimizations
   - Additional visualizations
   - Extended documentation and examples
   - GPU acceleration (optional)

This phased approach allows for early validation of the core concept while systematically building out the full feature set.
