"""Tests for handling learning convergence failures."""

from unittest.mock import patch

import numpy as np
import pytest

# Import test fixtures


# Define a custom exception for convergence failures
class MockConvergenceError(Exception):
    """Exception raised when the training process fails to converge."""


class MockConvergenceWarning(Warning):
    """Warning issued when the training process may have convergence issues."""


# Mock estimator that fails to converge
class FailingEstimator:
    """Mock estimator that fails to converge during training."""

    def __init__(self, function, param_ranges, independent_vars_sampling, fail_type="error"):
        """Initialize the estimator.

        Args:
            function: The function to fit
            param_ranges: Dictionary of parameter ranges
            independent_vars_sampling: Dictionary of sampling points
            fail_type: Type of failure ('error', 'warning', or 'silent')
        """
        self.function = function
        self.param_ranges = param_ranges
        self.independent_vars_sampling = independent_vars_sampling
        self.fail_type = fail_type

        self.is_trained = False

    def train(self, n_epochs=10, batch_size=32, learning_rate=0.001):
        """Train the estimator, but fail to converge."""
        # Simulate training iterations
        for epoch in range(n_epochs):
            # Simulate a training loss that doesn't improve
            loss = 1.0 - epoch * 0.01  # Very slow improvement

            # Check if learning is stalled
            if epoch > 5 and loss > 0.9:  # Not making good progress
                if self.fail_type == "error":
                    raise MockConvergenceError("Training failed to converge. Loss did not decrease sufficiently.")
                elif self.fail_type == "warning":
                    import warnings

                    warnings.warn(
                        "Training may not have converged. Loss is not decreasing sufficiently.",
                        MockConvergenceWarning,
                        stacklevel=2,
                    )
                    break
                else:  # silent failure
                    break

        # Mark as trained even if it failed silently
        if self.fail_type != "error":
            self.is_trained = True

        return {"loss": loss, "val_loss": loss * 1.1, "converged": False}

    def predict(self, x_data, y_data):
        """Predict parameters, returning poor estimates if training failed."""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")

        # Return poor parameter estimates (median of each range)
        params = {}
        for param_name, (min_val, max_val) in self.param_ranges.items():
            params[param_name] = (min_val + max_val) / 2

        return params


class TestLearningFailure:
    """Tests for handling learning convergence failures."""

    def test_convergence_error(self, set_random_seeds, difficult_function):
        """Test that a convergence error is properly raised."""
        # Define parameter ranges for the difficult function
        param_ranges = {param: (-10, 10) for param in "abcdefghij"}

        # Define sampling points
        independent_vars_sampling = {"x": np.linspace(0, 10, 100)}

        # Create estimator that will fail with an error
        estimator = FailingEstimator(
            difficult_function,
            param_ranges,
            independent_vars_sampling,
            fail_type="error",
        )

        # Training should raise a convergence error
        with pytest.raises(MockConvergenceError):
            estimator.train()

        # Verify the estimator is not trained
        assert not estimator.is_trained

    def test_convergence_warning(self, set_random_seeds, difficult_function):
        """Test that a convergence warning is properly issued."""
        # Define parameter ranges for the difficult function
        param_ranges = {param: (-10, 10) for param in "abcdefghij"}

        # Define sampling points
        independent_vars_sampling = {"x": np.linspace(0, 10, 100)}

        # Create estimator that will fail with a warning
        estimator = FailingEstimator(
            difficult_function,
            param_ranges,
            independent_vars_sampling,
            fail_type="warning",
        )

        # Training should issue a warning
        with pytest.warns(MockConvergenceWarning):
            training_result = estimator.train()

        # Verify the estimator is still marked as trained
        assert estimator.is_trained

        # Verify the training result indicates non-convergence
        assert not training_result.get("converged", True)

    def test_silent_failure_prediction_quality(self, set_random_seeds, difficult_function):
        """Test that predictions from a silently failed model are flagged as unreliable."""
        # Define parameter ranges for the difficult function
        param_ranges = {param: (-10, 10) for param in "abcdefghij"}

        # Define sampling points
        independent_vars_sampling = {"x": np.linspace(0, 10, 100)}

        # Create estimator that will fail silently
        estimator = FailingEstimator(
            difficult_function,
            param_ranges,
            independent_vars_sampling,
            fail_type="silent",
        )

        # Train the estimator (it will fail silently)
        training_result = estimator.train()

        # Verify the training result indicates non-convergence
        assert not training_result.get("converged", True)

        # Generate some test data
        x_data = np.linspace(0, 10, 50)
        y_data = difficult_function(x_data, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)

        # Get predictions
        predicted_params = estimator.predict(x_data, y_data)

        # Verify that all parameters have been estimated
        assert set(predicted_params.keys()) == set(param_ranges.keys())

        # In a real implementation, the ZeroGuess library would provide some indication
        # of prediction reliability. Here we just check that the parameters are within range.
        for param_name, value in predicted_params.items():
            min_val, max_val = param_ranges[param_name]
            assert min_val <= value <= max_val

    def test_recovery_from_failure(self, set_random_seeds, difficult_function):
        """Test recovery strategy after a convergence failure."""
        # Define parameter ranges for the difficult function
        param_ranges = {param: (-10, 10) for param in "abcdefghij"}

        # Define sampling points
        independent_vars_sampling = {"x": np.linspace(0, 10, 100)}

        # Create estimator that will fail with a warning
        estimator = FailingEstimator(
            difficult_function,
            param_ranges,
            independent_vars_sampling,
            fail_type="warning",
        )

        # Training with the default parameters will issue a warning
        with pytest.warns(MockConvergenceWarning):
            estimator.train()

        # Create a patched version that simulates successful training with modified parameters
        with patch.object(
            FailingEstimator,
            "train",
            return_value={"loss": 0.1, "val_loss": 0.2, "converged": True},
        ) as mock_train:
            # Simulate a retry with different parameters
            retry_result = estimator.train(n_epochs=100, batch_size=64, learning_rate=0.0001)

            # Verify that train was called with the new parameters
            mock_train.assert_called_once_with(n_epochs=100, batch_size=64, learning_rate=0.0001)

            # Verify that the retry succeeded
            assert retry_result.get("converged", False)
