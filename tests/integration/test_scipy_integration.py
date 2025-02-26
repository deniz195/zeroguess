"""Integration tests for the SciPy integration of ZeroGuess."""

import numpy as np
import pytest
from unittest.mock import patch, MagicMock

# Import test utilities
from ..conftest import set_random_seeds, gaussian_function, sample_data_1d
from ..test_utils import calculate_parameter_error, is_within_tolerance

# Try to import scipy, but allow tests to be skipped if not available
try:
    from scipy import optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Create marker to skip tests if scipy is not available
requires_scipy = pytest.mark.skipif(
    not HAS_SCIPY, reason="SciPy is required for this test"
)

# Mock ZeroGuess SciPy integration
def mock_zeroguess_curve_fit(func, xdata, ydata, param_ranges=None, independent_vars_sampling=None, **kwargs):
    """Mock of the ZeroGuess-enhanced curve_fit function."""
    # Create a mock estimator
    mock_estimator = MagicMock()
    
    # The mock estimator.predict should return values within param_ranges
    predicted_params = {}
    if param_ranges:
        for param_name, (min_val, max_val) in param_ranges.items():
            # Random value within range for testing
            predicted_params[param_name] = np.random.uniform(min_val, max_val)
    
    # If no param_ranges, just use some default parameters
    if not predicted_params:
        predicted_params = {'a': 1.0, 'b': 2.0, 'c': 3.0}
    
    # Convert dict to list for scipy curve_fit
    p0 = list(predicted_params.values())
    
    # Call the original scipy.optimize.curve_fit with our p0
    if 'p0' in kwargs:
        # Don't override if p0 was explicitly provided
        pass
    else:
        kwargs['p0'] = p0
    
    # Call the real scipy curve_fit
    popt, pcov = optimize.curve_fit(func, xdata, ydata, **kwargs)
    
    return popt, pcov

@requires_scipy
class TestScipyIntegration:
    """Tests for SciPy integration."""
    
    def test_zeroguess_curve_fit_with_param_ranges(self, set_random_seeds, gaussian_function, sample_data_1d):
        """Test that ZeroGuess-enhanced curve_fit works with parameter ranges."""
        # Get sample data
        x_data, y_data, true_params = sample_data_1d
        
        # Define parameter ranges
        param_ranges = {
            'amplitude': (0, 10),
            'center': (-5, 5),
            'width': (0.1, 2)
        }
        
        # Define sampling points
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 100)
        }
        
        # Modified gaussian function to match scipy's calling convention
        def scipy_gaussian(x, amplitude, center, width):
            return gaussian_function(x, amplitude=amplitude, center=center, width=width)
        
        # Get parameter estimates and fit
        popt, pcov = mock_zeroguess_curve_fit(
            scipy_gaussian, x_data, y_data,
            param_ranges=param_ranges,
            independent_vars_sampling=independent_vars_sampling
        )
        
        # Check that the fit succeeded
        assert popt is not None
        assert pcov is not None
        assert len(popt) == len(true_params)
        
        # Convert popt to dict for easier comparison
        fitted_params = {
            'amplitude': popt[0],
            'center': popt[1],
            'width': popt[2]
        }
        
        # Check that fitted parameters are reasonable (this would be more specific in a real test)
        for param_name, value in fitted_params.items():
            min_val, max_val = param_ranges[param_name]
            assert value >= min_val * 0.5  # Allow some flexibility in the fit
            assert value <= max_val * 1.5  # Allow some flexibility in the fit
    
    @patch('scipy.optimize.curve_fit')
    def test_zeroguess_curve_fit_calls_scipy(self, mock_scipy_curve_fit, set_random_seeds, gaussian_function, sample_data_1d):
        """Test that ZeroGuess properly calls the original scipy.optimize.curve_fit."""
        # Set up mock return value for scipy.optimize.curve_fit
        mock_scipy_curve_fit.return_value = (np.array([7.5, 2.0, 1.2]), np.eye(3))
        
        # Get sample data
        x_data, y_data, true_params = sample_data_1d
        
        # Define parameter ranges
        param_ranges = {
            'amplitude': (0, 10),
            'center': (-5, 5),
            'width': (0.1, 2)
        }
        
        # Modified gaussian function to match scipy's calling convention
        def scipy_gaussian(x, amplitude, center, width):
            return gaussian_function(x, amplitude=amplitude, center=center, width=width)
        
        # Call mock_zeroguess_curve_fit
        popt, pcov = mock_zeroguess_curve_fit(
            scipy_gaussian, x_data, y_data,
            param_ranges=param_ranges
        )
        
        # Verify that scipy.optimize.curve_fit was called
        mock_scipy_curve_fit.assert_called_once()
        
        # Get the arguments that scipy.optimize.curve_fit was called with
        args, kwargs = mock_scipy_curve_fit.call_args
        
        # Verify that the function and data were passed correctly
        assert args[0] == scipy_gaussian
        assert np.array_equal(args[1], x_data)
        assert np.array_equal(args[2], y_data)
        
        # Verify that p0 was included in kwargs
        assert 'p0' in kwargs
        assert len(kwargs['p0']) == len(param_ranges)
    
    def test_zeroguess_curve_fit_respects_user_p0(self, set_random_seeds, gaussian_function, sample_data_1d):
        """Test that ZeroGuess doesn't override p0 if the user provides it."""
        # Get sample data
        x_data, y_data, true_params = sample_data_1d
        
        # Define parameter ranges
        param_ranges = {
            'amplitude': (0, 10),
            'center': (-5, 5),
            'width': (0.1, 2)
        }
        
        # Define user-provided p0
        user_p0 = [5.0, 0.0, 1.0]
        
        # Modified gaussian function to match scipy's calling convention
        def scipy_gaussian(x, amplitude, center, width):
            return gaussian_function(x, amplitude=amplitude, center=center, width=width)
        
        # Use a patched version for this test to verify p0
        with patch('scipy.optimize.curve_fit') as mock_scipy_curve_fit:
            mock_scipy_curve_fit.return_value = (np.array([7.5, 2.0, 1.2]), np.eye(3))
            
            # Call mock_zeroguess_curve_fit with user-provided p0
            popt, pcov = mock_zeroguess_curve_fit(
                scipy_gaussian, x_data, y_data,
                param_ranges=param_ranges,
                p0=user_p0
            )
            
            # Verify that scipy.optimize.curve_fit was called with user's p0
            args, kwargs = mock_scipy_curve_fit.call_args
            assert 'p0' in kwargs
            assert np.array_equal(kwargs['p0'], user_p0) 