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
│   ├── architectures/   # Neural network architectures
│   │   ├── __init__.py
│   │   ├── base.py      # Base architecture interface
│   │   ├── mlp.py       # Multilayer perceptron architecture
│   │   ├── cnn.py       # Convolutional neural network architecture
│   │   ├── transformer.py # Transformer architecture (Future Work)
│   │   └── registry.py  # Architecture registry and selection
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
│   ├── metrics.py       # Training metrics and evaluation
│   └── benchmarking.py  # Architecture benchmarking utilities (Future Work)
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
  - Select and configure appropriate network architectures

#### Neural Network Architecture Components
- **Purpose**: Provide different neural network architectures optimized for various fitting scenarios
- **Responsibilities**:
  - Implement specific neural network architectures (MLP, CNN, Transformer, etc.)
  - Provide architecture-specific hyperparameter configurations
  - Register architectures with the architecture registry
  - Apply architecture-specific regularization and optimization techniques

#### Architecture Registry
- **Purpose**: Manage available neural network architectures
- **Responsibilities**:
  - Register and catalog available architectures
  - Provide a selection mechanism for choosing architectures
  - Manage default "best" architecture selection
  - Handle backward compatibility for code not specifying an architecture

#### Architecture Benchmarking
- **Purpose**: Evaluate and compare different architectures
- **Responsibilities**:
  - Measure performance metrics for different architectures on specific problems
   - Compare architecture performance across different function types
   - Generate reports on architecture performance

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
   - User optionally selects a neural network architecture and configuration
   - System validates inputs and configures data generation
   - Architecture-specific configurations are validated and defaults applied

2. **Data Generation Phase**:
   - System generates random parameter sets within ranges
   - System evaluates fitting function with these parameters
   - Training dataset is created from parameter-output pairs

3. **Architecture Initialization Phase**:
   - Selected architecture (or default) is instantiated
   - Architecture-specific components are configured
   - Model is built with appropriate input/output dimensions

4. **Training Phase**:
   - Neural network is trained with architecture-specific optimizations
   - Model is trained on synthetic data
   - Training metrics are monitored for convergence/issues
   - Architecture-specific regularization is applied

5. **Inference Phase**:
   - User provides experimental data
   - System preprocesses the data according to architecture requirements
   - Model predicts initial parameters
   - System validates prediction quality
   - Parameters are returned to user or injected into fitting workflow

6. **Benchmarking Phase (Optional)** *(Future Work)*:
   - Multiple architectures are instantiated and trained
   - Performance metrics are collected for each architecture
   - Results are compared and visualized
   - Recommendations are generated based on performance

7. **Persistence (Optional)**:
   - Trained model is serialized to disk with architecture information
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

# Create parameter estimator with specific architecture
estimator = zeroguess.create_estimator(
    function=gaussian,
    param_ranges={
        'amplitude': (0, 10),
        'center': (-5, 5),
        'width': (0.1, 2)
    },
    independent_vars_sampling={
        'x': x_sampling  # Sampling points for the independent variable
    },
    architecture='mlp',  # Select MLP architecture (optional, defaults to "best")
    architecture_params={  # Architecture-specific parameters (optional)
        'hidden_layers': [64, 128, 64],
        'activation': 'relu',
        'dropout_rate': 0.1
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

# Benchmark different architectures for this function (optional) (Future Work)
benchmark_results = zeroguess.benchmark_architectures(
    function=gaussian,
    param_ranges={
        'amplitude': (0, 10),
        'center': (-5, 5),
        'width': (0.1, 2)
    },
    independent_vars_sampling={
        'x': x_sampling
    },
    architectures=['mlp', 'cnn', 'transformer'],  # Architectures to benchmark
    metrics=['mse', 'parameter_error', 'training_time']  # Metrics to evaluate
)

# Print benchmark results
print(benchmark_results.summary())
# Or visualize the results
benchmark_results.plot()
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
    },
    architecture='cnn',  # Optional: Select CNN architecture
    architecture_params={  # Optional: Architecture-specific parameters
        'n_conv_layers': 2,
        'filters': [16, 32],
        'kernel_size': 3
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
    independent_vars_sampling={
        'x': x_sampling  # Sampling points for training
    },
    estimator_settings={
        'architecture': 'mlp',  # Optional: Select MLP architecture
        'architecture_params': {  # Optional: Architecture-specific parameters
            'hidden_layers': [64, 128, 64],
            'activation': 'relu'
        }
    }
)

# Set parameter bounds (will be automatically used for parameter estimation)
model.set_param_hint('amplitude', min=0, max=10)
model.set_param_hint('center', min=-5, max=5)
model.set_param_hint('width', min=0.1, max=2)

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
)

# Set parameter bounds (will be automatically used for parameter estimation)
model.set_param_hint('amplitude', min=0, max=10)
model.set_param_hint('center', min=-5, max=5)
model.set_param_hint('width', min=0.1, max=2)

# Standard lmfit workflow continues
result = model.fit(y_data, x=x_data)
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

# Note: Transformer architecture integration is planned for future work

## Neural Network Architectures

ZeroGuess supports multiple neural network architectures, each optimized for different types of curve fitting problems. The architecture selection feature allows users to choose the most appropriate architecture for their specific needs.

### Architecture Design Principles and Restrictions

#### Input Size Handling
- **Input Size Restriction**: All models/estimators only allow the input sizes they are trained against. This ensures consistency between training and inference phases and prevents unexpected behavior when prediction is performed on data with different dimensions than the training data.

#### Architecture Responsibility Pattern
- **"Tell, Don't Ask" Principle**: All architectures follow the same interface and are responsible for their own behavior. This encapsulation ensures that architecture-specific logic remains within the architecture implementation rather than in the estimator code, promoting cleaner separation of concerns and better maintainability.
- Each architecture is responsible for validating its own input requirements, handling its specific data transformations, and implementing appropriate forward/backward propagation logic.

### Hardware Acceleration

ZeroGuess automatically selects the optimal hardware acceleration for neural network training and inference based on availability. This ensures models train efficiently across different computing environments without manual configuration.

#### Device Selection Strategy
- **Automatic Selection**: During initialization, ZeroGuess determines the best available hardware in the following priority order:
  1. CUDA (NVIDIA GPUs)
  2. MPS (Apple Silicon)
  3. CPU (fallback option)

- **Manual Override**: For specialized use cases, the API provides options to manually specify the target hardware:
  ```python
  # Example of manual device selection (API to be implemented)
  estimator = NeuralNetworkEstimator(
      function=target_function,
      param_ranges=param_ranges,
      independent_vars_sampling=sampling,
      device="cpu"  # Force CPU execution even if GPU is available
  )
  ```

- **Supported Hardware Backends**:
  - **CUDA**: Acceleration for NVIDIA GPUs
  - **MPS**: Metal Performance Shaders for Apple Silicon (M1/M2/M3 chips)
  - **CPU**: Standard CPU execution for universal compatibility

The automatic hardware selection ensures optimal performance without user intervention while maintaining the flexibility to target specific hardware when needed.

### Architecture Types

#### 1. Multilayer Perceptron (MLP)
- **Description**: A standard feedforward neural network with fully connected layers
- **Strengths**: 
  - General-purpose architecture suitable for a wide range of functions
  - Good performance on smooth, continuous functions
  - Computationally efficient for small to medium-sized problems
- **Configuration Parameters**:
  - `hidden_layers`: List of integers specifying the size of each hidden layer
  - `activation`: Activation function to use (relu, tanh, sigmoid, etc.)
  - `dropout_rate`: Dropout rate for regularization (0.0-1.0)
- **Best For**: General-purpose use, simple unimodal functions, default architecture

#### 2. Convolutional Neural Network (CNN)
- **Description**: Architecture using convolutional layers to capture local patterns in data
- **Strengths**:
  - Excels at capturing local patterns and periodicities
  - Robust to input size variations
  - Can handle higher-dimensional inputs efficiently
- **Configuration Parameters**:
  - `n_conv_layers`: Number of convolutional layers
  - `filters`: List of integers specifying the number of filters in each layer
  - `kernel_size`: Size of the convolutional kernels
  - `pool_size`: Size of the pooling windows (if used)
- **Best For**: Oscillatory functions, functions with repeating patterns, multi-peak functions

#### 3. Transformer Architecture *(Not Currently Implemented - Future Work)*
- **Description**: Based on the transformer architecture with self-attention mechanisms
- **Strengths**:
  - Excels at capturing global relationships in data
  - Can handle complex dependencies between different regions of the input
  - Superior performance on multi-modal and complex functions
- **Configuration Parameters**:
  - `n_heads`: Number of attention heads
  - `n_layers`: Number of transformer layers
  - `dim_feedforward`: Dimension of the feedforward network in each transformer layer
  - `dropout`: Dropout rate for attention layers
- **Best For**: Complex, multi-modal functions, functions with long-range dependencies

### Default "Universal" Architecture

The default architecture (selected when no specific architecture is specified) is a carefully tuned MLP configuration designed to perform well across a wide range of common functions. This architecture is referred to as the "universal" architecture and features:

- A pyramid-like structure of hidden layers that gradually reduces dimensionality
- Batch normalization for improved training stability
- Residual connections to help with gradient flow
- Adaptive learning rate scheduling

### Architecture Selection Mechanism

The architecture selection is implemented through a registry pattern:

1. **Architecture Registry**: A central registry that maintains a catalog of available architectures
2. **Factory Function**: Creates the appropriate architecture instance based on the selected type
3. **Configuration Validation**: Validates and applies defaults for architecture-specific parameters

### Benchmarking System *(Not Currently Implemented - Future Work)*

The benchmarking system allows users to compare different architectures on their specific problems:

1. **Metrics Collection**: Gathers performance metrics for each architecture
   - Mean squared error (MSE) on validation data
   - Parameter estimation error
   - Training time
   - Inference time
   - Memory usage

2. **Comparison Visualization**: Generates visualizations comparing architectures
   - Bar charts for quantitative metrics
   - Radar plots for multi-dimensional comparison
   - Training curve comparisons

3. **Results Analysis**: Provides recommendations based on benchmarking results
   - Identifies the best architecture for the specific problem
   - Highlights trade-offs between different architectures
   - Suggests potential hyperparameter improvements

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
   - Python 3.10+ (target 3.10.7)
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

2. **Phase 2**: Architecture Selection
   - Base architecture interface
   - MLP architecture implementation
   - Architecture registry
   - Factory function enhancements
   - API updates for architecture selection
   - Backward compatibility layer

3. **Phase 3**: Advanced Architectures *(Partial Implementation)*
   - CNN architecture implementation
   - ~~Transformer architecture implementation~~ *(Future Work)*
   - Architecture-specific hyperparameter handling
   - ~~Architecture benchmarking system~~ *(Future Work)*

4. **Phase 4**: Verification and Optimization
   - Add verification examples as unit tests
   - Add verification tests for each architecture
   - Comprehensive error handling
   - Performance optimizations

5. **Phase 5**: Enhanced functionality and Future Extensions
   - Architecture benchmarking system
   - Transformer architecture implementation
   - Improved architecture selection heuristics
   - Model persistence
   - Additional visualizations
   - Extended documentation and examples
   - GPU acceleration (optional)

This phased approach allows for early validation of the core concept while systematically building out the full feature set.

## ZeroGuess Functions Submodule

### Overview

The functions submodule provides a collection of pre-defined fitting functions commonly used in curve fitting applications. Each function is encapsulated in a class that provides the function implementation, default parameter ranges, and default independent variable sampling points.

This submodule is designed to be:
- **Independent**: No dependencies on core ZeroGuess functionality
- **User-friendly**: Easy to use with sensible defaults
- **Extensible**: Easy to add new functions
- **Educational**: Includes parameter descriptions

### Module Structure

```
zeroguess/
├── functions/
│   ├── __init__.py           # Exports all functions
│   ├── base.py               # Base class definition
│   ├── standard.py           # Common fitting functions
│   └── utils.py              # Utility functions (noise, etc.)
```

### Base Function Class

The `FittingFunction` base class defines the interface that all function implementations must follow:

```python
class FittingFunction:
    """Base class for fitting functions."""
    
    @property
    def name(self) -> str:
        """Return the name of the function."""
        pass
        
    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges."""
        pass
        
    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        pass
        
    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        pass
        
    def __call__(self, *args, **kwargs):
        """Evaluate the function with the given parameters."""
        pass
        
    def generate_data(self, params: Optional[Dict[str, float]] = None, 
                     indep_vars: Optional[Dict[str, np.ndarray]] = None) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Generate data for the function with the given parameters."""
        pass
```

### Standard Functions

The following standard fitting functions will be included:

1. **Gaussian Function**: A bell-shaped curve defined by amplitude, center, and width
2. **Multi-Peak Gaussian**: A sum of multiple Gaussian functions
3. **Damped Sine Wave**: Oscillatory function with exponential decay
4. **Linear Function**: Simple linear relationship with slope and intercept

### Noise Utility

The module includes a noise utility for adding Gaussian noise to generated data:

```python
def add_gaussian_noise(data: np.ndarray, sigma: float = 0.1, relative: bool = True) -> np.ndarray:
    """Add Gaussian noise to data."""
    pass
```

### Usage Example

```python
from zeroguess.functions import GaussianFunction

# Create a Gaussian function instance
gaussian = GaussianFunction()

# View parameter descriptions
print(gaussian.param_descriptions)

# Generate data with default parameters
indep_vars, dependent_var = gaussian.generate_data()

# Generate data with custom parameters
params = {'amplitude': 2.0, 'center': 0.5, 'width': 1.5}
indep_vars, dependent_var = gaussian.generate_data(params)

# Add noise to the data
from zeroguess.functions import add_gaussian_noise
noisy_dependent_var = add_gaussian_noise(dependent_var, sigma=0.05)
```

### Integration with ZeroGuess (Optional)

While designed to be independent, these function classes can be easily used with the core ZeroGuess functionality:

```python
from zeroguess import create_estimator
from zeroguess.functions import GaussianFunction

# Create function instance
gaussian = GaussianFunction()

# Create estimator using function information
estimator = create_estimator(
    function=gaussian,
    param_ranges=gaussian.param_ranges,
    independent_vars_sampling=gaussian.default_independent_vars
)
```

### Extension

Users can create custom fitting functions by subclassing `FittingFunction`:

```python
from zeroguess.functions import FittingFunction
import numpy as np
from typing import Dict, Tuple

class CustomFunction(FittingFunction):
    """Custom fitting function implementation."""
    
    @property
    def name(self) -> str:
        return "custom_function"
        
    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            'param1': (0.0, 10.0),
            'param2': (-5.0, 5.0)
        }
        
    @property
    def param_descriptions(self) -> Dict[str, str]:
        return {
            'param1': 'Description of parameter 1',
            'param2': 'Description of parameter 2'
        }
        
    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        return {
            'x': np.linspace(-10, 10, 100)
        }
        
    def __call__(self, x, param1, param2):
        """Evaluate the function with the given parameters."""
        return param1 * x + param2
```

This architecture provides a clean, user-friendly way to work with common fitting functions while maintaining flexibility and extensibility.

## Release Process

This section outlines the process for releasing new versions of ZeroGuess to PyPI.

### Versioning Strategy

ZeroGuess follows [Semantic Versioning (SemVer)](https://semver.org/):

- **MAJOR** version increments for incompatible API changes
- **MINOR** version increments for backwards-compatible functionality additions
- **PATCH** version increments for backwards-compatible bug fixes

Version numbers are stored in `zeroguess/__init__.py` as `__version__`.

### Release Preparation

1. **Update Documentation**:
   - Ensure all new features and changes are documented
   - Update README.md with any new examples if applicable
   - Update CHANGELOG.md with a detailed list of changes

2. **Run Quality Checks**:
   - Ensure all tests pass: `pytest tests/`
   - Check code coverage: `pytest --cov=zeroguess tests/`
   - Run benchmarks to ensure performance hasn't regressed: `python examples/run_benchmark_1.py all`
   - Run code quality checks: `./scripts/quality.py`

3. **Version Bump**:
   - Update version in `zeroguess/__init__.py`
   - Update version in `pyproject.toml`
   - Commit the version bump: `git commit -am "Bump version to X.Y.Z"`
   - Create a tag: `git tag vX.Y.Z`
   - Push changes and tag: `git push && git push --tags`

### Building and Publishing

1. **Clean Build Environment**:
   ```bash
   # Remove any previous builds
   rm -rf build/ dist/ *.egg-info/
   ```

2. **Build the Package**:
   ```bash
   # Install build tools if needed
   pip install --upgrade build twine

   # Build both wheel and source distribution
   python -m build
   ```

3. **Test the Package**:
   ```bash
   # Validate the package
   twine check dist/*
   
   # Test in a clean environment
   pip install --force-reinstall dist/*.whl
   python -c "import zeroguess; print(zeroguess.__version__)"
   ```

4. **Publish to Test PyPI** (Optional):
   ```bash
   # Upload to Test PyPI first
   twine upload --repository-url https://test.pypi.org/legacy/ dist/*
   
   # Verify installation from Test PyPI
   pip install --index-url https://test.pypi.org/simple/ zeroguess
   ```

5. **Publish to PyPI**:
   ```bash
   # Upload to PyPI
   twine upload dist/*
   ```

### Post-Release Activities

1. **Create GitHub Release**:
   - Go to the repository's releases page on GitHub
   - Create a new release using the tag
   - Copy the relevant section from CHANGELOG.md as the release description

2. **Announce Release**:
   - Post announcement in relevant channels (GitHub Discussions, social media, etc.)
   - Update documentation site if separate from repository

3. **Development Continuation**:
   - Update version in `zeroguess/__init__.py` to next development version (e.g., "X.Y+1.0-dev")
   - Create milestone for next version if applicable

### Automation

The release process can be partially automated using GitHub Actions. A workflow file at `.github/workflows/release.yml` can handle the building and publishing steps:

With this workflow, publishing to PyPI is triggered automatically when a new GitHub release is created.
