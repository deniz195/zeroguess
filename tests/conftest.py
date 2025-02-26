"""Common test fixtures and utilities for ZeroGuess tests."""

import os
import random
import numpy as np
import pytest
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Skip tests that require torch if it's not installed
requires_torch = pytest.mark.skipif(
    not HAS_TORCH, reason="PyTorch is required for this test"
)

@pytest.fixture(scope="function")
def set_random_seeds():
    """Set random seeds for reproducibility in tests.
    
    This fixture ensures that tests using random number generation
    are reproducible by setting fixed seeds. After the test completes,
    it resets settings to allow for non-deterministic operations.
    """
    # Store original states if needed
    original_np_state = np.random.get_state()
    
    # Set seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    
    # Set PyTorch seeds if available
    if HAS_TORCH:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    yield
    
    # Reset to original state after test
    np.random.set_state(original_np_state)
    
    # Reset PyTorch settings
    if HAS_TORCH:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


@pytest.fixture
def gaussian_function():
    """Return a basic Gaussian function for testing."""
    def gaussian(x, amplitude, center, width):
        return amplitude * np.exp(-(x - center)**2 / (2 * width**2))
    return gaussian


@pytest.fixture
def sample_data_1d():
    """Generate a 1D sample dataset with controlled parameters.
    
    Returns:
        tuple: (x_data, y_data, true_params) containing the independent variable,
               dependent variable, and the true parameters used to generate the data.
    """
    # Fixed parameters for reproducibility
    true_params = {'amplitude': 7.5, 'center': 2.0, 'width': 1.2}
    
    # Generate data points
    x_data = np.linspace(-10, 10, 50)
    
    # Get the Gaussian function
    def gaussian(x, amplitude, center, width):
        return amplitude * np.exp(-(x - center)**2 / (2 * width**2))
    
    # Generate clean data
    y_clean = gaussian(x_data, **true_params)
    
    # Add noise
    np.random.seed(42)  # Set seed for reproducibility
    noise = np.random.normal(0, 0.1, size=len(x_data))
    y_data = y_clean + noise
    
    return x_data, y_data, true_params


@pytest.fixture
def multi_peak_gaussian_function():
    """Return a multi-peak Gaussian function for testing edge cases."""
    def multi_peak_gaussian(x, amp1, center1, width1, amp2, center2, width2):
        return (amp1 * np.exp(-(x - center1)**2 / (2 * width1**2)) + 
                amp2 * np.exp(-(x - center2)**2 / (2 * width2**2)))
    return multi_peak_gaussian


@pytest.fixture
def difficult_function():
    """Return a pathologically difficult function for testing edge cases."""
    def function(x, a, b, c, d, e, f, g, h, i, j):
        # Function with many parameters and complex behavior
        # Note: the random component is meant to make this function particularly
        # difficult to fit and should trigger convergence warnings/errors
        np.random.seed(42)  # Fixed seed for reproducibility within the function
        return (a * np.sin(b * x + c) + 
                d * np.cos(e * x + f) + 
                g * x**2 + h * x + i + 
                j * np.random.random(size=len(x) if hasattr(x, '__len__') else 1))
    return function 