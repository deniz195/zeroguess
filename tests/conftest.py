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
def sample_data_multi_peak():
    """Generate a multi-peak Gaussian dataset with controlled parameters.
    
    Returns:
        tuple: (x_data, y_data, true_params) containing the independent variable,
               dependent variable, and the true parameters used to generate the data.
    """
    # Fixed parameters for reproducibility
    true_params = {
        'amp1': 5.0, 'center1': -2.0, 'width1': 1.0,
        'amp2': 3.0, 'center2': 2.0, 'width2': 0.8
    }
    
    # Generate data points with higher sampling density for better peak resolution
    x_data = np.linspace(-10, 10, 100)
    
    # Get the multi-peak Gaussian function
    def multi_peak_gaussian(x, amp1, center1, width1, amp2, center2, width2):
        return (amp1 * np.exp(-(x - center1)**2 / (2 * width1**2)) + 
                amp2 * np.exp(-(x - center2)**2 / (2 * width2**2)))
    
    # Generate clean data
    y_clean = multi_peak_gaussian(x_data, **true_params)
    
    # Add noise
    np.random.seed(42)  # Set seed for reproducibility
    noise = np.random.normal(0, 0.15, size=len(x_data))  # Slightly higher noise
    y_data = y_clean + noise
    
    return x_data, y_data, true_params


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


@pytest.fixture
def sigmoid_function():
    """Return a sigmoid/logistic function for testing."""
    def sigmoid(x, amplitude, center, rate):
        return amplitude / (1 + np.exp(-rate * (x - center)))
    return sigmoid


@pytest.fixture
def sample_data_sigmoid():
    """Generate a sigmoid dataset with controlled parameters.
    
    Returns:
        tuple: (x_data, y_data, true_params) containing the independent variable,
               dependent variable, and the true parameters used to generate the data.
    """
    # Fixed parameters for reproducibility
    true_params = {'amplitude': 5.0, 'center': 0.0, 'rate': 2.0}
    
    # Generate data points
    x_data = np.linspace(-10, 10, 100)
    
    # Get the sigmoid function
    def sigmoid(x, amplitude, center, rate):
        return amplitude / (1 + np.exp(-rate * (x - center)))
    
    # Generate clean data
    y_clean = sigmoid(x_data, **true_params)
    
    # Add noise
    np.random.seed(42)  # Set seed for reproducibility
    noise = np.random.normal(0, 0.2, size=len(x_data))
    y_data = y_clean + noise
    
    return x_data, y_data, true_params


@pytest.fixture
def double_sigmoid_function():
    """Return a double sigmoid function for testing more complex sigmoid cases."""
    def double_sigmoid(x, amp1, center1, rate1, amp2, center2, rate2):
        return (amp1 / (1 + np.exp(-rate1 * (x - center1))) + 
                amp2 / (1 + np.exp(-rate2 * (x - center2))))
    return double_sigmoid


@pytest.fixture
def sample_data_double_sigmoid():
    """Generate a double sigmoid dataset with controlled parameters.
    
    Returns:
        tuple: (x_data, y_data, true_params) containing the independent variable,
               dependent variable, and the true parameters used to generate the data.
    """
    # Fixed parameters for reproducibility
    true_params = {
        'amp1': 3.0, 'center1': -2.0, 'rate1': 1.5,
        'amp2': 2.0, 'center2': 2.0, 'rate2': 1.0
    }
    
    # Generate data points with higher sampling density
    x_data = np.linspace(-10, 10, 100)
    
    # Get the double sigmoid function
    def double_sigmoid(x, amp1, center1, rate1, amp2, center2, rate2):
        return (amp1 / (1 + np.exp(-rate1 * (x - center1))) + 
                amp2 / (1 + np.exp(-rate2 * (x - center2))))
    
    # Generate clean data
    y_clean = double_sigmoid(x_data, **true_params)
    
    # Add noise
    np.random.seed(42)  # Set seed for reproducibility
    noise = np.random.normal(0, 0.15, size=len(x_data))
    y_data = y_clean + noise
    
    return x_data, y_data, true_params


@pytest.fixture
def multimodal_function():
    """Return a multimodal function with local minima for testing.
    
    This function is defined as:
    f(x) = a1 * sin(a2 * x) + a3 * cos(a4 * x + a5)
    
    It creates a complex landscape with multiple local minima, testing
    the ability of the system to handle multimodal optimization problems.
    """
    def multimodal(x, a1, a2, a3, a4, a5):
        return a1 * np.sin(a2 * x) + a3 * np.cos(a4 * x + a5)
    return multimodal


@pytest.fixture
def sample_data_multimodal():
    """Generate a multimodal dataset with controlled parameters.
    
    Returns:
        tuple: (x_data, y_data, true_params) containing the independent variable,
               dependent variable, and the true parameters used to generate the data.
    """
    # Fixed parameters for reproducibility
    true_params = {
        'a1': 2.5,   # Amplitude of sine component
        'a2': 0.8,   # Frequency of sine component
        'a3': 3.0,   # Amplitude of cosine component
        'a4': 1.5,   # Frequency of cosine component
        'a5': 0.5    # Phase shift of cosine component
    }
    
    # Generate data points with higher sampling density
    x_data = np.linspace(-10, 10, 120)
    
    # Get the multimodal function
    def multimodal(x, a1, a2, a3, a4, a5):
        return a1 * np.sin(a2 * x) + a3 * np.cos(a4 * x + a5)
    
    # Generate clean data
    y_clean = multimodal(x_data, **true_params)
    
    # Add noise
    np.random.seed(42)  # Set seed for reproducibility
    noise = np.random.normal(0, 0.2, size=len(x_data))
    y_data = y_clean + noise
    
    return x_data, y_data, true_params 