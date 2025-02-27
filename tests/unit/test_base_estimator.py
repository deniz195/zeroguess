"""
Unit tests for base estimator functionality.
"""

import unittest
from abc import ABC

import numpy as np
import pytest

from zeroguess import create_estimator
from zeroguess.estimators.base import BaseEstimator

# Import the test fixtures

# Try to import the BaseEstimator class (legacy code, can be removed if BaseEstimator is always available)
try:
    from zeroguess.estimators.base import BaseEstimator

    HAS_ZEROGUESS = True
except ImportError:
    HAS_ZEROGUESS = False

    # Create a mock BaseEstimator for the test to run even if zeroguess is not available
    class BaseEstimator(ABC):
        pass


# Create marker to skip tests if zeroguess is not available
requires_zeroguess = pytest.mark.skipif(not HAS_ZEROGUESS, reason="ZeroGuess is required for this test")


# Create a minimal implementation of BaseEstimator for testing
class MockEstimator(BaseEstimator):
    """Mock implementation of BaseEstimator for testing purposes."""

    def __init__(self, function, param_ranges, independent_vars_sampling, **kwargs):
        """Custom initializer that validates inputs first to avoid AttributeError."""
        # Validate inputs before calling super().__init__
        if not callable(function):
            raise TypeError("Function must be callable")

        if not isinstance(param_ranges, dict) or not param_ranges:
            raise ValueError("param_ranges must be a non-empty dictionary")

        for param_name, param_range in param_ranges.items():
            if not isinstance(param_range, tuple) or len(param_range) != 2:
                raise ValueError(f"Range for parameter {param_name} must be a tuple of (min, max)")
            if param_range[0] >= param_range[1]:
                raise ValueError(f"Min value must be less than max value for parameter {param_name}")

        if not isinstance(independent_vars_sampling, dict) or not independent_vars_sampling:
            raise ValueError("independent_vars_sampling must be a non-empty dictionary")

        for var_name, sampling in independent_vars_sampling.items():
            if not isinstance(sampling, np.ndarray):
                raise ValueError(f"Sampling points for {var_name} must be a numpy array")
            if sampling.size == 0:
                raise ValueError(f"Sampling points for {var_name} must not be empty")

        # Now it's safe to call the parent constructor
        super().__init__(function, param_ranges, independent_vars_sampling, **kwargs)

    def train(self, n_samples=1000, **kwargs):
        """Mock implementation of train method."""
        self.is_trained = True
        return {"loss": 0.1}

    def predict(self, x_data, y_data):
        """Mock implementation of predict method."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Return middle of range for all parameters
        return {name: (min_val + max_val) / 2 for name, (min_val, max_val) in self.param_ranges.items()}

    def save(self, path):
        """Mock implementation of save method."""

    @classmethod
    def load(cls, path):
        """Mock implementation of load method."""

        # Return a dummy instance
        def dummy_function(x):
            return x

        param_ranges = {"a": (0, 1)}
        independent_vars_sampling = {"x": np.array([0, 1])}
        return cls(dummy_function, param_ranges, independent_vars_sampling)


@requires_zeroguess
class TestBaseEstimator:
    """Tests for the BaseEstimator abstract base class."""

    def test_initialization_with_valid_inputs(self, gaussian_function):
        """Test that the estimator initializes correctly with valid inputs."""
        # Define valid parameter ranges
        param_ranges = {"amplitude": (0, 10), "center": (-5, 5), "width": (0.1, 2)}

        # Define valid sampling points
        independent_vars_sampling = {"x": np.linspace(-10, 10, 100)}

        # Initialize estimator
        estimator = MockEstimator(gaussian_function, param_ranges, independent_vars_sampling)

        # Check that parameters are stored correctly
        assert estimator.param_ranges == param_ranges
        assert estimator.independent_vars_sampling == independent_vars_sampling
        assert set(estimator.param_names) == set(param_ranges.keys())
        assert set(estimator.independent_var_names) == set(independent_vars_sampling.keys())
        assert not estimator.is_trained

    def test_function_validation(self):
        """Test that the estimator validates the function is callable."""
        # Define valid parameter ranges and sampling points
        param_ranges = {"a": (0, 1)}
        independent_vars_sampling = {"x": np.array([0, 1])}

        # Try to initialize with non-callable function
        with pytest.raises(TypeError, match="Function must be callable"):
            MockEstimator("not_callable", param_ranges, independent_vars_sampling)

    def test_param_ranges_validation(self, gaussian_function):
        """Test that the estimator validates parameter ranges."""
        # Define valid sampling points
        independent_vars_sampling = {"x": np.linspace(-10, 10, 100)}

        # Test with empty param_ranges
        with pytest.raises(ValueError, match="param_ranges must be a non-empty dictionary"):
            MockEstimator(gaussian_function, {}, independent_vars_sampling)

        # Test with non-dictionary param_ranges
        with pytest.raises(ValueError, match="param_ranges must be a non-empty dictionary"):
            MockEstimator(gaussian_function, "not_a_dict", independent_vars_sampling)

        # Test with invalid range format
        invalid_ranges = {"amplitude": [0, 10]}  # List instead of tuple
        with pytest.raises(ValueError, match="Range for parameter amplitude must be a tuple"):
            MockEstimator(gaussian_function, invalid_ranges, independent_vars_sampling)

        # Test with min > max
        invalid_ranges = {"amplitude": (10, 0)}  # Min > Max
        with pytest.raises(ValueError, match="Min value must be less than max value"):
            MockEstimator(gaussian_function, invalid_ranges, independent_vars_sampling)

    def test_independent_vars_validation(self, gaussian_function):
        """Test that the estimator validates independent variables sampling."""
        # Define valid parameter ranges
        param_ranges = {"amplitude": (0, 10), "center": (-5, 5), "width": (0.1, 2)}

        # Test with empty independent_vars_sampling
        with pytest.raises(ValueError, match="independent_vars_sampling must be a non-empty dictionary"):
            MockEstimator(gaussian_function, param_ranges, {})

        # Test with non-dictionary independent_vars_sampling
        with pytest.raises(ValueError, match="independent_vars_sampling must be a non-empty dictionary"):
            MockEstimator(gaussian_function, param_ranges, "not_a_dict")

        # Test with non-numpy array sampling
        invalid_sampling = {"x": [0, 1, 2]}  # List instead of numpy array
        with pytest.raises(ValueError, match="Sampling points for x must be a numpy array"):
            MockEstimator(gaussian_function, param_ranges, invalid_sampling)

        # Test with empty sampling array
        invalid_sampling = {"x": np.array([])}  # Empty array
        with pytest.raises(ValueError, match="Sampling points for x must not be empty"):
            MockEstimator(gaussian_function, param_ranges, invalid_sampling)

    def test_training_interface(self, gaussian_function):
        """Test that the training method works correctly."""
        # Initialize with valid parameters
        param_ranges = {"amplitude": (0, 10), "center": (-5, 5), "width": (0.1, 2)}
        independent_vars_sampling = {"x": np.linspace(-10, 10, 100)}

        estimator = MockEstimator(gaussian_function, param_ranges, independent_vars_sampling)

        # Check that the estimator is not initially trained
        assert not estimator.is_trained

        # Train the estimator
        training_result = estimator.train(n_samples=500)

        # Check that training completed
        assert estimator.is_trained
        assert "loss" in training_result

    def test_prediction_interface(self, gaussian_function, sample_data_1d):
        """Test that the prediction method works correctly."""
        # Get sample data
        x_data, y_data, true_params = sample_data_1d

        # Initialize with valid parameters
        param_ranges = {"amplitude": (0, 10), "center": (-5, 5), "width": (0.1, 2)}
        independent_vars_sampling = {"x": np.linspace(-10, 10, 100)}

        estimator = MockEstimator(gaussian_function, param_ranges, independent_vars_sampling)

        # Prediction should raise error because model is not trained
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            estimator.predict(x_data, y_data)

        # Train the estimator
        estimator.train()

        # Now prediction should work
        predicted_params = estimator.predict(x_data, y_data)

        # Check that prediction has the correct structure
        assert set(predicted_params.keys()) == set(param_ranges.keys())

        # In our mock implementation, values should be the middle of the range
        for param_name, (min_val, max_val) in param_ranges.items():
            expected_value = (min_val + max_val) / 2
            assert predicted_params[param_name] == expected_value

    def test_abstract_methods(self):
        """Test that abstract methods must be implemented by subclasses."""
        if not HAS_ZEROGUESS:
            pytest.skip("ZeroGuess is required for this test")

        # Define a new class that inherits from BaseEstimator but doesn't implement all abstract methods
        class IncompleteEstimator(BaseEstimator):
            """Incomplete implementation of BaseEstimator."""

            # Only implement some methods
            def train(self, n_samples=1000, **kwargs):
                pass

            def predict(self, x_data, y_data):
                pass

            # Missing save and load methods

        # Define a dummy function that will pass validation
        def dummy_function(x, a):
            return a * x

        # Define valid parameters
        param_ranges = {"a": (0, 1)}
        independent_vars_sampling = {"x": np.array([0, 1])}

        # Trying to instantiate this class should raise TypeError
        with pytest.raises(TypeError):
            IncompleteEstimator(dummy_function, param_ranges, independent_vars_sampling)


class TestBaseEstimator(unittest.TestCase):
    """Tests for base estimator functionality."""

    def setUp(self):
        """Set up common test data."""

        # Define a simple function for testing
        def linear_function(x, a, b):
            return a * x + b

        self.function = linear_function
        self.param_ranges = {"a": (0.5, 5.0), "b": (-2.0, 2.0)}
        self.independent_vars_sampling = {"x": np.linspace(-10, 10, 100)}

    def test_create_estimator_basic(self):
        """Test creating an estimator with basic parameters."""
        estimator = create_estimator(
            function=self.function,
            param_ranges=self.param_ranges,
            independent_vars_sampling=self.independent_vars_sampling,
        )
        self.assertIsNotNone(estimator)

    def test_estimator_param_validation(self):
        """Test that estimator properly validates function parameters."""
        # Test with empty param range
        with self.assertRaises(ValueError):
            create_estimator(
                function=self.function,
                param_ranges={},  # Empty parameter range
                independent_vars_sampling=self.independent_vars_sampling,
            )

        # Test with invalid param range format
        with self.assertRaises(ValueError):
            create_estimator(
                function=self.function,
                param_ranges={"a": (5.0, 0.5), "b": (-2.0, 2.0)},  # Min > Max (invalid)
                independent_vars_sampling=self.independent_vars_sampling,
            )

        # Test with empty independent vars sampling
        with self.assertRaises(ValueError):
            create_estimator(
                function=self.function,
                param_ranges=self.param_ranges,
                independent_vars_sampling={},  # Empty independent vars
            )

    def test_estimator_architecture_param_validation(self):
        """Test that estimator properly validates architecture parameters."""
        # Test with invalid architecture parameters for MLP
        with self.assertRaises(ValueError):
            create_estimator(
                function=self.function,
                param_ranges=self.param_ranges,
                independent_vars_sampling=self.independent_vars_sampling,
                architecture="mlp",
                architecture_params={"invalid_mlp_param": 123},  # Parameter that doesn't exist for MLP
            )

        # Test with invalid architecture parameters for CNN
        with self.assertRaises(ValueError):
            create_estimator(
                function=self.function,
                param_ranges=self.param_ranges,
                independent_vars_sampling=self.independent_vars_sampling,
                architecture="cnn",
                architecture_params={"unknown_cnn_param": 456},  # Parameter that doesn't exist for CNN
            )

        # Test that valid architecture parameters work correctly
        try:
            # Valid MLP parameters
            create_estimator(
                function=self.function,
                param_ranges=self.param_ranges,
                independent_vars_sampling=self.independent_vars_sampling,
                architecture="mlp",
                architecture_params={
                    "hidden_layers": [32, 64, 32],
                    "dropout_rate": 0.2,
                },
            )

            # Valid CNN parameters
            create_estimator(
                function=self.function,
                param_ranges=self.param_ranges,
                independent_vars_sampling=self.independent_vars_sampling,
                architecture="cnn",
                architecture_params={"n_conv_layers": 3, "filters": [16, 32, 64]},
            )
        except Exception as e:
            self.fail(f"Creating estimator with valid architecture parameters raised {type(e).__name__}: {e}")


if __name__ == "__main__":
    unittest.main()
