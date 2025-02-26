"""Integration tests for the SciPy integration of ZeroGuess."""

import numpy as np
import pytest
from unittest.mock import patch

# Import test utilities
from ..conftest import set_random_seeds, gaussian_function, sample_data_1d
from ..test_utils import calculate_parameter_error, is_within_tolerance

# Try to import scipy, but allow tests to be skipped if not available
try:
    from scipy import optimize
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# Import the real ZeroGuess scipy integration
try:
    from zeroguess.integration import scipy_integration
    HAS_ZEROGUESS = True
except ImportError:
    HAS_ZEROGUESS = False

# Create marker to skip tests if scipy is not available
requires_scipy = pytest.mark.skipif(
    not HAS_SCIPY, reason="SciPy is required for this test"
)

# Create marker to skip tests if zeroguess is not available
requires_zeroguess = pytest.mark.skipif(
    not HAS_ZEROGUESS, reason="ZeroGuess is required for this test"
)

@requires_scipy
@requires_zeroguess
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
        
        # Use the same x_data for sampling and fitting to avoid NotImplementedError
        # in the current implementation of NeuralNetworkEstimator.predict
        independent_vars_sampling = {
            'x': x_data
        }
        
        # Modified gaussian function to match scipy's calling convention
        def scipy_gaussian(x, amplitude, center, width):
            return gaussian_function(x, amplitude=amplitude, center=center, width=width)
        
        # Get parameter estimates and fit using real ZeroGuess scipy integration
        popt, pcov = scipy_integration.curve_fit(
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
        
        # Calculate parameter errors compared to true values
        errors = calculate_parameter_error(fitted_params, true_params)
        
        # Check that errors are within reasonable tolerance for a real ML model
        # Using a higher tolerance as we're using a real neural network estimator
        for param_name, error in errors.items():
            assert error <= 2.0, f"Error for {param_name} too high: {error:.2f}"
        
        # Calculate curve fit quality
        fitted_y = scipy_gaussian(x_data, *popt)
        rmse = np.sqrt(np.mean((y_data - fitted_y)**2))
        
        # Check that RMSE is reasonable
        assert rmse < 1.0, f"RMSE too high: {rmse:.2f}"
    
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
        
        # Use the same x_data for sampling to avoid NotImplementedError
        independent_vars_sampling = {
            'x': x_data
        }
        
        # Modified gaussian function to match scipy's calling convention
        def scipy_gaussian(x, amplitude, center, width):
            return gaussian_function(x, amplitude=amplitude, center=center, width=width)
        
        # Call real ZeroGuess scipy integration
        popt, pcov = scipy_integration.curve_fit(
            scipy_gaussian, x_data, y_data,
            param_ranges=param_ranges,
            independent_vars_sampling=independent_vars_sampling
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
            
            # Call real ZeroGuess scipy integration with user-provided p0
            popt, pcov = scipy_integration.curve_fit(
                scipy_gaussian, x_data, y_data,
                param_ranges=param_ranges,
                p0=user_p0
            )
            
            # Verify that scipy.optimize.curve_fit was called with user's p0
            args, kwargs = mock_scipy_curve_fit.call_args
            assert 'p0' in kwargs
            assert np.array_equal(kwargs['p0'], user_p0)
            
    def test_zeroguess_curve_fit_without_param_ranges(self, set_random_seeds, gaussian_function, sample_data_1d):
        """Test that ZeroGuess falls back to standard curve_fit when no param_ranges are provided."""
        # Get sample data
        x_data, y_data, true_params = sample_data_1d
        
        # Modified gaussian function to match scipy's calling convention
        def scipy_gaussian(x, amplitude, center, width):
            return gaussian_function(x, amplitude=amplitude, center=center, width=width)
        
        # Use a patched version to verify that p0 is not in kwargs
        with patch('scipy.optimize.curve_fit') as mock_scipy_curve_fit:
            mock_scipy_curve_fit.return_value = (np.array([7.5, 2.0, 1.2]), np.eye(3))
            
            # Call without param_ranges
            popt, pcov = scipy_integration.curve_fit(
                scipy_gaussian, x_data, y_data
            )
            
            # Verify that scipy.optimize.curve_fit was called without p0
            args, kwargs = mock_scipy_curve_fit.call_args
            assert 'p0' not in kwargs, "p0 should not be provided when param_ranges is not provided" 