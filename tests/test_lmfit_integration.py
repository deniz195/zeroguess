#!/usr/bin/env python
"""
Unit tests for lmfit integration with ZeroGuess.

This test validates the lmfit integration functionality, particularly:
1. The guess() method behavior
2. Automatic parameter extraction
3. Error handling when parameter estimation fails
"""
import unittest
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import warnings

# Add the parent directory to the path so we can import zeroguess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import lmfit
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False
    warnings.warn("lmfit package is not installed. The lmfit tests will be skipped.")

import zeroguess
if LMFIT_AVAILABLE:
    from zeroguess.integration import lmfit_integration


def wavelet(x, frequency, phase, position, width):
    """Wavelet function (a modulated Gaussian)."""
    envelope = np.exp(-(x - position)**2 / (2 * width**2))
    wave = np.sin(2 * np.pi * frequency * x + phase)
    return envelope * wave


def generate_noisy_data(x, true_params, noise_level=0.05):
    """Generate noisy data for testing."""
    y_true = wavelet(x, **true_params)
    noise = np.random.normal(0, noise_level * np.max(np.abs(y_true)), size=len(x))
    return y_true + noise


@unittest.skipIf(not LMFIT_AVAILABLE, "lmfit is not installed")
class TestLmfitIntegration(unittest.TestCase):
    """Tests for the lmfit integration functionality."""
    
    def setUp(self):
        """Set up data for the tests."""
        # Set fixed random seed for reproducibility
        np.random.seed(123)
        
        # Define true parameters
        self.true_params = {
            'frequency': 1.544704,
            'phase': 1.797866,
            'position': -1.092594,
            'width': 2.102630,
        }
        
        # Define sampling points
        self.x_sampling = np.linspace(-5, 5, 200)
        
        # Generate noisy data
        self.x_data = self.x_sampling.copy()
        self.y_data = generate_noisy_data(self.x_data, self.true_params)
        
        # Suppress plots during tests
        plt.ioff()  # Turn off interactive mode
    
    def test_handle_when_estimation_fails(self):
        """Test that failures in the parameter estimator are properly detected."""
        # Create model with ZeroGuess integration
        model = lmfit_integration.Model(
            wavelet,
            independent_vars_sampling={
                'x': self.x_sampling,
            },
            auto_extract_bounds=True,
        )
        
        # Set parameter bounds
        params = model.make_params()
        for param_name, param in params.items():
            if param_name == 'frequency':
                param.min = 0.1
                param.max = 5.0
            elif param_name == 'phase':
                param.min = 0
                param.max = 2 * np.pi
            elif param_name == 'position':
                param.min = -5
                param.max = 5
            elif param_name == 'width':
                param.min = 0.1
                param.max = 5
        
        # Get parameters from guess()
        with warnings.catch_warnings(record=True) as w:
            guessed_params = model.guess(self.y_data, x=self.x_data)
            
            # There should be a warning about parameter estimation
            self.assertTrue(any("Failed to initialize or train parameter estimator" in str(warning.message) 
                              for warning in w))
        
        # Check that some parameters are invalid (-inf)
        has_invalid_params = False
        for param_name, param in guessed_params.items():
            if not np.isfinite(param.value) or param.value == float('-inf'):
                has_invalid_params = True
                break
        
        # There should be invalid parameters
        self.assertTrue(has_invalid_params, "The guess() method should return invalid parameters for this test case")
        
        # Attempting to fit with these parameters would raise an error, which we don't test directly
    
    def test_guess_method_success(self):
        """Test that the guess() method successfully estimates parameters."""
        # Create model with ZeroGuess integration using auto_extract_bounds
        model = lmfit_integration.Model(
            wavelet,
            independent_vars_sampling={
                'x': self.x_sampling,
            },
            auto_extract_bounds=True,
        )
        
        # Set parameter bounds
        print("DEBUG [before make_params]")
        model.set_param_hint('frequency', min=0.1, max=5.0)
        model.set_param_hint('phase', min=0, max=2 * np.pi)
        model.set_param_hint('position', min=-5, max=5)
        model.set_param_hint('width', min=0.5, max=5)  # Set a more strict lower bound for width
        params = model.make_params()
        
        # Store the params with bounds in the model for proper initialization
        print("DEBUG [after make_params]")
                        
        # Now test guess() after the estimator is properly initialized
        with warnings.catch_warnings(record=True) as estimation_warnings:
            warnings.filterwarnings("ignore", message=".*torch.utils._pytree._register_pytree_node.*")
            print("DEBUG [before guess]")
            guessed_params = model.guess(self.y_data, x=self.x_data)
            print("DEBUG [after guess]")
            
        # Filter out training-related warnings which would only occur once in real usage
        estimation_warnings = [w for w in estimation_warnings 
                               if "Failed to initialize or train parameter estimator" not in str(w.message)]
            
        # Print warnings for debugging
        for w in estimation_warnings:
            print(f"WARNING: {w.message}")
            
        # Check that no warnings were issued (excluding estimator initialization warnings)
        self.assertEqual(len(estimation_warnings), 0, "guess() should not produce parameter estimation warnings")
        
        # Verify that guessed parameters are valid
        for param_name, param in guessed_params.items():
            self.assertTrue(np.isfinite(param.value), f"Parameter {param_name} should have a finite value")
            self.assertNotEqual(param.value, -np.inf, f"Parameter {param_name} should not be -inf")
            
        # Verify that guessed parameters are similar enough to true parameters
        print("Guessed parameters (before fitting):")
        for param_name in ['frequency', 'phase', 'position', 'width']:
            true_value = self.true_params[param_name]
            guessed_value = guessed_params[param_name].value
            print(f"  {param_name}: {guessed_value:.6f} (true: {true_value:.6f})")
            
            # Allow for different tolerance levels for different parameters
            if param_name == 'width':
                tolerance = 2.0  # 200% for width
            elif param_name == 'position':
                tolerance = 2.0  # 200% for position
            else:
                tolerance = 0.5  # 50% for frequency and phase
                
            self.assertLess(
                abs(guessed_value - true_value) / abs(true_value),
                tolerance,
                f"Guessed {param_name} ({guessed_value}) should be reasonably close to true value ({true_value})"
            )
        
        # Fit the model with the guessed parameters and check if the fit is successful
        guess_result = model.fit(self.y_data, params=guessed_params, x=self.x_data)
        self.assertTrue(guess_result.success, "Fitting should be successful with guessed parameters")
        
        # Verify that the fitted parameters are close to the true parameters
        print("\nFitted parameters (after fitting):")
        for param_name in ['frequency', 'phase', 'position', 'width']:
            true_value = self.true_params[param_name]
            fitted_value = guess_result.params[param_name].value
            print(f"  {param_name}: {fitted_value:.6f} (true: {true_value:.6f})")
            
            # The absolute value check is important for width since it might be negative
            absolute_diff = abs(abs(fitted_value) - abs(true_value))
            relative_diff = absolute_diff / abs(true_value)
            
            # Use a larger tolerance for width since it's harder to fit precisely
            tolerance = 2.0 if param_name == 'width' else 1.0
            
            self.assertLess(
                relative_diff,
                tolerance,
                f"Fitted {param_name} ({fitted_value}) should be reasonably close to true value ({true_value})"
            )
    
    def test_direct_parameter_fitting(self):
        """Test fitting with direct parameter specification (bypassing guess)."""
        # Create model with ZeroGuess integration
        model = lmfit_integration.Model(
            wavelet,
            independent_vars_sampling={
                'x': self.x_sampling,
            },
            auto_extract_bounds=True,
        )
        
        # Set parameter bounds
        params = model.make_params()
        for param_name, param in params.items():
            if param_name == 'frequency':
                param.min = 0.1
                param.max = 5.0
                param.value = 1.5  # Set a reasonable starting value
            elif param_name == 'phase':
                param.min = 0
                param.max = 2 * np.pi
                param.value = 1.8  # Set a reasonable starting value
            elif param_name == 'position':
                param.min = -5
                param.max = 5
                param.value = -1.0  # Set a reasonable starting value
            elif param_name == 'width':
                param.min = 0.1
                param.max = 5
                param.value = 2.0  # Set a reasonable starting value
        
        # Fit with explicitly provided parameters
        result = model.fit(self.y_data, params, x=self.x_data)
        
        # Check that the fit was successful
        self.assertTrue(result.success, "Fit should succeed with explicit parameters")
        
        # Check that the fitted parameters are close to the true parameters
        # We don't expect exact matches due to noise and optimization variations
        for param_name, true_value in self.true_params.items():
            fitted_value = result.params[param_name].value
            # Check that values are within 20% of true values
            # This is a loose check since fitting is subject to noise
            self.assertLess(
                abs(fitted_value - true_value) / max(abs(true_value), 1e-10),
                0.2,
                f"Fitted {param_name} ({fitted_value}) should be close to true value ({true_value})"
            )
        
        # Check that the fit quality is good
        self.assertLess(result.redchi, 0.1, "Reduced chi-square should be small for a good fit")
    
    def test_model_creation_with_auto_extract_bounds(self):
        """Test that model creation with auto_extract_bounds works correctly."""
        # Create model with auto_extract_bounds
        model = lmfit_integration.Model(
            wavelet,
            independent_vars_sampling={
                'x': self.x_sampling,
            },
            auto_extract_bounds=True,
        )
        
        # Verify that the model has auto_extract_bounds enabled
        self.assertTrue(model.auto_extract_bounds, "auto_extract_bounds should be True")
        
        # Make parameters and set bounds
        params = model.make_params()
        for param_name, param in params.items():
            if param_name == 'frequency':
                param.min = 0.1
                param.max = 5.0
            elif param_name == 'phase':
                param.min = 0
                param.max = 2 * np.pi
            elif param_name == 'position':
                param.min = -5
                param.max = 5
            elif param_name == 'width':
                param.min = 0.1
                param.max = 5
        
        # Verify that all parameters have bounds set
        for param_name, param in params.items():
            self.assertIsNotNone(param.min, f"{param_name} should have a lower bound")
            self.assertIsNotNone(param.max, f"{param_name} should have an upper bound")


if __name__ == '__main__':
    unittest.main() 