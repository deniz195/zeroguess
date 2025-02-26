# ZeroGuess: Testing Strategy

This document outlines the testing strategy for ZeroGuess, focusing on verification through end-to-end tests with specific examples, addressing the non-deterministic nature of ML training, and ensuring proper handling of edge cases.

## Testing Philosophy

The testing strategy for ZeroGuess is guided by the following principles:

1. **Verification-focused**: Tests should verify that the library fulfills its promised functionality and meets performance requirements.
2. **End-to-end testing**: Tests should validate the entire workflow from parameter specification to model training and prediction.
3. **Reproducibility**: Despite the non-deterministic nature of ML, tests should be reasonably reproducible.
4. **Edge case handling**: Tests should verify proper error handling and warnings for difficult scenarios.
5. **Integration validation**: Tests should verify seamless integration with SciPy and lmfit libraries.

## Testing Framework

We will use pytest as our testing framework with the following additional packages:

- `pytest-cov`: For measuring test coverage
- `pytest-benchmark`: For performance benchmarking
- `pytest-xdist`: For parallel test execution

## Benchmark Functions

### Common Curve Fitting Functions

1. **Gaussian**:
   ```python
   def gaussian(x, amplitude, center, width):
       return amplitude * np.exp(-(x - center)**2 / (2 * width**2))
   ```

2. **Sigmoid/Logistic**:
   ```python
   def sigmoid(x, amplitude, center, rate):
       return amplitude / (1 + np.exp(-rate * (x - center)))
   ```

3. **Exponential Decay**:
   ```python
   def exp_decay(x, amplitude, decay_rate):
       return amplitude * np.exp(-decay_rate * x)
   ```

4. **Sine Wave**:
   ```python
   def sine_wave(x, amplitude, frequency, phase):
       return amplitude * np.sin(frequency * x + phase)
   ```

5. **Linear Function**:
   ```python
   def linear(x, slope, intercept):
       return slope * x + intercept
   ```

6. **Polynomial (Quadratic)**:
   ```python
   def quadratic(x, a, b, c):
       return a * x**2 + b * x + c
   ```

7. **Lorentzian**:
   ```python
   def lorentzian(x, amplitude, center, width):
       return amplitude * width**2 / ((x - center)**2 + width**2)
   ```

### Challenging Curve Fitting Functions

1. **Multi-peak Gaussian**:
   ```python
   def multi_peak_gaussian(x, amp1, center1, width1, amp2, center2, width2):
       return (amp1 * np.exp(-(x - center1)**2 / (2 * width1**2)) + 
               amp2 * np.exp(-(x - center2)**2 / (2 * width2**2)))
   ```

2. **Damped Oscillation**:
   ```python
   def damped_oscillation(x, amplitude, frequency, decay, phase):
       return amplitude * np.exp(-decay * x) * np.sin(frequency * x + phase)
   ```

3. **Voigt Profile** (convolution of Gaussian and Lorentzian):
   ```python
   def voigt_profile(x, amplitude, center, sigma, gamma):
       from scipy.special import voigt_profile
       return amplitude * voigt_profile(x - center, sigma, gamma)
   ```

4. **Skewed Gaussian**:
   ```python
   def skewed_gaussian(x, amplitude, center, width, skew):
       t = (x - center) / width
       return amplitude * np.exp(-t**2/2) * (1 + erf(skew * t / np.sqrt(2)))
   ```

5. **Functions with Local Minima**:
   ```python
   def multimodal(x, a1, a2, a3, a4, a5):
       return a1 * np.sin(a2 * x) + a3 * np.cos(a4 * x + a5)
   ```

## Handling Non-Deterministic Behavior

ML training processes are inherently non-deterministic due to random weight initialization, stochastic optimization, and other factors. Here are options for handling this in tests:

### Options:

1. **Fixed Seeds**: Set fixed random seeds for all random number generators (numpy, torch, etc.) at the beginning of each test.
   - Pros: Makes tests more reproducible
   - Cons: May hide issues that only occur with certain initializations

2. **Statistical Testing**: Run multiple training sessions and apply statistical tests to the results.
   - Pros: More robust assessment of true performance
   - Cons: Increases test runtime significantly

3. **Higher Margins**: Accept non-determinism by setting wider acceptance thresholds.
   - Pros: Simpler implementation
   - Cons: May allow some regressions to pass tests

4. **Deterministic Operations**: Force PyTorch to use deterministic algorithms where possible.
   - Pros: Increases reproducibility
   - Cons: May affect performance and not all operations can be made deterministic

5. **Saved Model Validation**: Instead of testing the training process, train models in advance and save them, then test against these saved models.
   - Pros: Eliminates training variability from tests
   - Cons: Doesn't test the training process itself

### Recommended Approach: Hybrid Strategy

We recommend a hybrid approach combining multiple strategies:

1. **For Unit Tests**: Use fixed seeds and deterministic operations to ensure reproducibility.

2. **For Integration Tests**: Use fixed seeds but with slightly relaxed acceptance criteria.

3. **For End-to-End Tests**: Use a combination of:
   - Fixed seeds for basic reproducibility
   - Statistical assertions that allow for some variance (e.g., "accuracy > 90%" rather than "accuracy == 95%")
   - Occasional full statistical testing (e.g., in nightly builds but not on every commit)

4. **For CI/CD Pipeline**: Include a mix of fast deterministic tests and more thorough statistical tests.

Implementation Example:
```python
@pytest.fixture(scope="function")
def set_random_seeds():
    # Set seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    yield
    # Reset to non-deterministic mode after test
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def test_gaussian_fitting(set_random_seeds):
    # Test with fixed seeds
    # ...
```

## Performance Thresholds

The following are placeholder performance thresholds that will be refined during testing:

1. **Accuracy Thresholds**:
   - Parameter estimation error < X% for high SNR (≥ 10) data
   - Parameter estimation error < Y% for medium SNR (5-10) data
   - Parameter estimation error < Z% for low SNR (2-5) data

2. **Training Performance**:
   - Training time < T seconds for standard functions (2-3 parameters)
   - Training time < U seconds for complex functions (4-6 parameters)
   - Training time < V seconds for very complex functions (7-10 parameters)

3. **Inference Performance**:
   - Inference time < W seconds per prediction
   - Memory usage < M MB during inference

4. **Convergence Metrics**:
   - Training loss should decrease by at least P% after Q epochs
   - Validation loss should not increase for more than R consecutive epochs

## Test Types

### 1. Unit Tests

Unit tests will focus on individual components of the ZeroGuess library:

- Data generation functions
- Model architecture components
- Training utilities
- IO operations

### 2. Integration Tests

Integration tests will verify the interaction between components:

- Data generation to model training workflow
- Model training to prediction workflow
- Integration with SciPy
- Integration with lmfit

### 3. End-to-End Tests

End-to-end tests will verify the complete workflow:

- Function specification → data generation → model training → parameter prediction → curve fitting
- Tests with both synthetic and real-world datasets
- Tests with varying noise levels and outliers

### 4. Edge Case Tests

Specific tests for edge cases and failure modes:

- **Bad Parameter Estimations**:
  - Functions with significant parameter correlation
  - Functions with redundant parameters
  - Parameters with vastly different scales
  - Parameters with narrow valid ranges

- **Learning Convergence Failures**:
  - Functions with multiple local minima
  - Highly nonlinear functions
  - Functions with discontinuities
  - Very high-dimensional parameter spaces

## Test Structure

Each test will follow this general structure:

```python
def test_[function]_[scenario]():
    # 1. Setup
    estimator = create_estimator_for_test(...)
    
    # 2. Training (if applicable)
    estimator.train()
    
    # 3. Generate test data
    x_test, y_test = generate_test_data(...)
    
    # 4. Make predictions
    params = estimator.predict(x_test, y_test)
    
    # 5. Validate predictions
    validate_predictions(params, expected_params, tolerance)
    
    # 6. Verify integration with fitting libraries (optional)
    result = curve_fit(function, x_test, y_test, p0=params)
    validate_fitting_result(result)
```

## Example Test Cases

### Basic Function Test

```python
def test_gaussian_estimation():
    """Test parameter estimation for a Gaussian function."""
    # Define function
    def gaussian(x, amplitude, center, width):
        return amplitude * np.exp(-(x - center)**2 / (2 * width**2))
    
    # Create estimator
    estimator = zeroguess.create_estimator(
        function=gaussian,
        param_ranges={
            'amplitude': (0, 10),
            'center': (-5, 5),
            'width': (0.1, 2)
        },
        independent_vars_sampling={
            'x': np.linspace(-10, 10, 100)
        }
    )
    
    # Train estimator
    estimator.train()
    
    # Generate test data with known parameters
    true_params = {'amplitude': 7.5, 'center': 2.0, 'width': 1.2}
    x_test = np.linspace(-10, 10, 50)
    y_test = gaussian(x_test, **true_params) + np.random.normal(0, 0.1, size=len(x_test))
    
    # Predict parameters
    estimated_params = estimator.predict(x_test, y_test)
    
    # Validate each parameter is within tolerance
    for param_name, true_value in true_params.items():
        estimated_value = estimated_params[param_name]
        relative_error = abs((estimated_value - true_value) / true_value)
        assert relative_error < 0.1, f"Parameter {param_name} estimation error too high: {relative_error:.2%}"
```

### Edge Case Test

```python
def test_learning_convergence_failure():
    """Test proper handling of learning convergence failures."""
    # Define a pathologically difficult function
    def difficult_function(x, a, b, c, d, e, f, g, h, i, j):
        # Function with many parameters and complex behavior
        return a*np.sin(b*x+c) + d*np.cos(e*x+f) + g*x**2 + h*x + i + j*np.random.random(len(x))
    
    # Try to create and train an estimator
    estimator = zeroguess.create_estimator(
        function=difficult_function,
        param_ranges={param: (-10, 10) for param in 'abcdefghij'},
        independent_vars_sampling={'x': np.linspace(0, 10, 100)}
    )
    
    # Training should issue a warning or exception
    with pytest.warns(zeroguess.ConvergenceWarning):
        try:
            estimator.train()
        except zeroguess.ConvergenceError as e:
            # Verify the error message is informative
            assert "convergence" in str(e).lower()
```

## Test Data Management

### Synthetic Data

Tests will primarily use synthetically generated data with controlled properties:

- Clean data (no noise)
- Data with Gaussian noise of varying SNR
- Data with outliers
- Data with missing values
- Data with varying density of sampling points

### Reference Datasets

A small set of reference datasets will be maintained for regression testing:

- Datasets that have caused issues in the past
- Datasets representing edge cases
- Real-world example datasets from common applications

## CI/CD Integration

Tests will be integrated into the CI/CD pipeline:

1. **On every commit**: Run fast tests (unit tests, basic integration tests)
2. **On PR merge**: Run full test suite
3. **Nightly**: Run extended tests including statistical validations and benchmarks

## Success Criteria

The test suite will be considered successful if:

1. It achieves high code coverage (target: >90% for core components)
2. It validates all key functionality of ZeroGuess
3. It properly detects regressions in performance or functionality
4. It provides clear feedback when tests fail

## Implementation Plan

1. **Phase 1**: Basic test infrastructure and unit tests
   - Set up pytest configuration
   - Implement test utilities
   - Create basic unit tests for all components

2. **Phase 2**: Integration tests
   - Test interactions between components
   - Verify library integrations

3. **Phase 3**: End-to-end tests and benchmarks
   - Create comprehensive examples
   - Implement performance benchmarks
   - Test with realistic scenarios

4. **Phase 4**: Edge case testing and robustness
   - Test failure modes
   - Test boundary conditions
   - Test with difficult functions 