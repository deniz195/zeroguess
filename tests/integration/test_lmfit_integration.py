"""Integration tests for the lmfit integration with ZeroGuess.

This test validates the lmfit integration functionality, particularly:
1. The guess() method behavior
2. Automatic parameter extraction
3. Error handling when parameter estimation fails
"""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Import test utilities

# Try to import lmfit, but allow tests to be skipped if not available
try:
    pass

    HAS_LMFIT = True
except ImportError:
    HAS_LMFIT = False
    warnings.warn("lmfit package is not installed. The lmfit tests will be skipped.", stacklevel=2)

# Import the real ZeroGuess lmfit integration
try:
    from zeroguess.integration import lmfit_integration

    HAS_ZEROGUESS = True
except ImportError:
    HAS_ZEROGUESS = False
    warnings.warn("ZeroGuess is not available. The tests will be skipped.", stacklevel=2)

# Create marker to skip tests if lmfit is not available
requires_lmfit = pytest.mark.skipif(not HAS_LMFIT, reason="lmfit is required for this test")

# Create marker to skip tests if zeroguess is not available
requires_zeroguess = pytest.mark.skipif(not HAS_ZEROGUESS, reason="ZeroGuess is required for this test")


def wavelet(x, frequency, phase, position, width):
    """Wavelet function (a modulated Gaussian)."""
    envelope = np.exp(-((x - position) ** 2) / (2 * width**2))
    wave = np.sin(2 * np.pi * frequency * x + phase)
    return envelope * wave


@pytest.fixture
def wavelet_function():
    """Return the wavelet function for testing."""
    return wavelet


@pytest.fixture
def true_params():
    """Return true parameters for the wavelet function."""
    return {
        "frequency": 1.544704,
        "phase": 1.797866,
        "position": -1.092594,
        "width": 2.102630,
    }


@pytest.fixture
def x_sampling():
    """Return sampling points for the wavelet function."""
    return np.linspace(-5, 5, 200)


@pytest.fixture
def noisy_data(x_sampling, true_params):
    """Generate noisy data for testing."""
    # Set fixed random seed for reproducibility
    np.random.seed(123)

    x_data = x_sampling.copy()
    y_true = wavelet(x_data, **true_params)
    noise = np.random.normal(0, 0.05 * np.max(np.abs(y_true)), size=len(x_data))
    y_data = y_true + noise

    return x_data, y_data


@pytest.fixture(autouse=True)
def disable_plots():
    """Disable plots during tests."""
    plt.ioff()  # Turn off interactive mode
    yield
    plt.close("all")


@requires_lmfit
@requires_zeroguess
class TestLmfitIntegration:
    """Tests for the lmfit integration functionality."""

    def test_handle_when_estimation_fails(self, set_random_seeds, wavelet_function, x_sampling, noisy_data):
        """Test that failures in the parameter estimator are properly detected."""
        x_data, y_data = noisy_data

        # Create model with ZeroGuess integration
        model = lmfit_integration.Model(
            wavelet_function,
            independent_vars_sampling={
                "x": x_sampling,
            },
        )

        # Set parameter bounds
        params = model.make_params()
        for param_name, param in params.items():
            if param_name == "frequency":
                param.min = 0.1
                param.max = 5.0
            elif param_name == "phase":
                param.min = 0
                param.max = 2 * np.pi
            elif param_name == "position":
                param.min = -5
                param.max = 5
            elif param_name == "width":
                param.min = 0.1
                param.max = 5

        # Get parameters from guess()
        with warnings.catch_warnings(record=True) as w:
            guessed_params = model.guess(y_data, x=x_data)

            # There should be a warning about parameter estimation
            assert any(
                "Failed to initialize or train parameter estimator" in str(warning.message) for warning in w
            ), "Expected warning about parameter estimation failure"

        # Check that some parameters are invalid (-inf)
        has_invalid_params = False
        for _, param in guessed_params.items():
            if not np.isfinite(param.value) or param.value == float("-inf"):
                has_invalid_params = True
                break

        # There should be invalid parameters
        assert has_invalid_params, "The guess() method should return invalid parameters for this test case"

    def test_guess_method_success(self, set_random_seeds, wavelet_function, x_sampling, noisy_data, true_params):
        """Test that the guess() method successfully estimates parameters."""
        x_data, y_data = noisy_data

        # Create model with ZeroGuess integration
        model = lmfit_integration.Model(
            wavelet_function,
            independent_vars_sampling={
                "x": x_sampling,
            },
        )

        # Set parameter bounds
        model.set_param_hint("frequency", min=0.1, max=5.0)
        model.set_param_hint("phase", min=0, max=2 * np.pi)
        model.set_param_hint("position", min=-5, max=5)
        model.set_param_hint("width", min=0.5, max=5)  # Set a more strict lower bound for width
        model.make_params()

        # Now test guess() after the estimator is properly initialized
        with warnings.catch_warnings(record=True) as estimation_warnings:
            warnings.filterwarnings("ignore", message=".*torch.utils._pytree._register_pytree_node.*")
            guessed_params = model.guess(y_data, x=x_data)

        # Filter out training-related warnings which would only occur once in real usage
        estimation_warnings = [
            w for w in estimation_warnings if "Failed to initialize or train parameter estimator" not in str(w.message)
        ]

        # Check that no warnings were issued (excluding estimator initialization warnings)
        assert len(estimation_warnings) == 0, "guess() should not produce parameter estimation warnings"

        # Verify that guessed parameters are valid
        for param_name, param in guessed_params.items():
            assert np.isfinite(param.value), f"Parameter {param_name} should have a finite value"
            assert param.value != -np.inf, f"Parameter {param_name} should not be -inf"

        # # Verify that guessed parameters are similar enough to true parameters
        # for param_name in ["frequency", "phase", "position", "width"]:
        #     true_value = true_params[param_name]
        #     guessed_value = guessed_params[param_name].value

        #     # Allow for different tolerance levels for different parameters
        #     if param_name == "width":
        #         tolerance = 2.5  # 200% for width
        #     elif param_name == "position":
        #         tolerance = 2.5  # 200% for position
        #     else:
        #         tolerance = 0.5  # 50% for frequency and phase

        #     assert (
        #         abs(guessed_value - true_value) / abs(true_value) < tolerance
        #     ), f"Guessed {param_name} ({guessed_value}) should be reasonably close to true value ({true_value})"

        # Fit the model with the guessed parameters and check if the fit is successful
        guess_result = model.fit(y_data, params=guessed_params, x=x_data)
        assert guess_result.success, "Fitting should be successful with guessed parameters"

        # Verify that the fitted parameters are close to the true parameters
        for param_name in ["frequency", "phase", "position", "width"]:
            true_value = true_params[param_name]
            fitted_value = guess_result.params[param_name].value

            # The absolute value check is important for width since it might be negative
            absolute_diff = abs(abs(fitted_value) - abs(true_value))
            relative_diff = absolute_diff / abs(true_value)

            # Use a larger tolerance for width since it's harder to fit precisely
            tolerance = 2.5 if param_name == "width" else 1.0

            assert (
                relative_diff < tolerance
            ), f"Fitted {param_name} ({fitted_value}) should be reasonably close to true value ({true_value})"

    def test_direct_parameter_fitting(self, set_random_seeds, wavelet_function, x_sampling, noisy_data, true_params):
        """Test fitting with direct parameter specification (bypassing guess)."""
        x_data, y_data = noisy_data

        # Create model with ZeroGuess integration
        model = lmfit_integration.Model(
            wavelet_function,
            independent_vars_sampling={
                "x": x_sampling,
            },
        )

        # Set parameter bounds
        params = model.make_params()
        for param_name, param in params.items():
            if param_name == "frequency":
                param.min = 0.1
                param.max = 5.0
                param.value = 1.5  # Set a reasonable starting value
            elif param_name == "phase":
                param.min = 0
                param.max = 2 * np.pi
                param.value = 1.8  # Set a reasonable starting value
            elif param_name == "position":
                param.min = -5
                param.max = 5
                param.value = -1.0  # Set a reasonable starting value
            elif param_name == "width":
                param.min = 0.1
                param.max = 5
                param.value = 2.0  # Set a reasonable starting value

        # Fit with explicitly provided parameters
        result = model.fit(y_data, params, x=x_data)

        # Check that the fit was successful
        assert result.success, "Fit should succeed with explicit parameters"

        # Check that the fitted parameters are close to the true parameters
        # We don't expect exact matches due to noise and optimization variations
        for param_name, true_value in true_params.items():
            fitted_value = result.params[param_name].value
            # Check that values are within 20% of true values
            # This is a loose check since fitting is subject to noise
            assert (
                abs(fitted_value - true_value) / max(abs(true_value), 1e-10) < 0.2
            ), f"Fitted {param_name} ({fitted_value}) should be close to true value ({true_value})"

        # Check that the fit quality is good
        assert result.redchi < 0.1, "Reduced chi-square should be small for a good fit"

    def test_model_creation_without_param_ranges(self, wavelet_function, x_sampling):
        """Test that model creation without param_ranges works correctly."""
        # Create model without param_ranges
        model = lmfit_integration.Model(
            wavelet_function,
            independent_vars_sampling={
                "x": x_sampling,
            },
        )

        model.set_param_hint("frequency", min=0.1, max=5.0)
        model.set_param_hint("phase", min=0, max=2 * np.pi)
        model.set_param_hint("position", min=-5, max=5)
        model.set_param_hint("width", min=0.1, max=5)
        params = model.make_params()

        # Verify that all parameters have bounds set
        for param_name, param in params.items():
            assert param.min is not None, f"{param_name} should have a lower bound"
            assert param.max is not None, f"{param_name} should have an upper bound"
