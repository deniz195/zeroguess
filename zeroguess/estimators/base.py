"""
Base parameter estimator interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np


class BaseEstimator(ABC):
    """Base class for parameter estimators.

    This abstract class defines the interface that all parameter estimators must implement.
    Concrete subclasses should override the train and predict methods.
    """

    def __init__(
        self,
        function: Optional[Callable],
        param_ranges: Dict[str, Tuple[float, float]],
        independent_vars_sampling: Dict[str, np.ndarray],
        **kwargs,
    ):
        """Initialize the parameter estimator.

        Args:
            function: The curve fitting target function
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            independent_vars_sampling: Dictionary mapping independent variable names
                to arrays of sampling points
            **kwargs: Additional keyword arguments
        """
        self.function = function
        self.param_ranges = param_ranges
        self.independent_vars_sampling = independent_vars_sampling
        self.param_names = list(param_ranges.keys())
        self.independent_var_names = list(independent_vars_sampling.keys())
        self.is_trained = False

        # Validate inputs
        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate the input parameters."""
        if self.function is not None and not callable(self.function):
            raise TypeError("Function must be callable")

        if not isinstance(self.param_ranges, dict) or not self.param_ranges:
            raise ValueError("Parameter ranges must be a non-empty dictionary")

        for param_name, (min_val, max_val) in self.param_ranges.items():
            if min_val >= max_val:
                raise ValueError(f"Min value must be less than max value for parameter {param_name}")

        if not isinstance(self.independent_vars_sampling, dict) or not self.independent_vars_sampling:
            raise ValueError("Independent variable sampling must be a non-empty dictionary")

        for var_name, sampling in self.independent_vars_sampling.items():
            if not isinstance(sampling, np.ndarray):
                raise TypeError(f"Sampling for {var_name} must be a numpy array")
            if sampling.size == 0:
                raise ValueError(f"Sampling for {var_name} must be non-empty")

    @abstractmethod
    def train(self, **kwargs) -> Dict[str, Any]:
        """Train the parameter estimator.

        Args:
            **kwargs: Training-specific arguments

        Returns:
            Dictionary containing training results
        """

    @abstractmethod
    def predict(self, *args, **kwargs) -> Dict[str, float]:
        """Predict initial parameters for a function.

        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Dictionary mapping parameter names to predicted values
        """

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: Path to save the model
        """

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseEstimator":
        """Load a trained model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded estimator instance
        """
