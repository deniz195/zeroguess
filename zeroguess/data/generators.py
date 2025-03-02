"""
Synthetic data generation for training parameter estimators.
"""

import inspect
from typing import Callable, Dict, Tuple

import numpy as np


class SyntheticDataGenerator:
    """Generator for synthetic training data for parameter estimation models."""

    def __init__(
        self,
        function: Callable,
        param_ranges: Dict[str, Tuple[float, float]],
        independent_vars_sampling: Dict[str, np.ndarray],
    ):
        """Initialize the synthetic data generator.

        Args:
            function: The curve fitting target function
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            independent_vars_sampling: Dictionary mapping independent variable names
                to arrays of sampling points
        """
        self.function = function
        self.param_ranges = param_ranges
        self.independent_vars_sampling = independent_vars_sampling
        self.param_names = list(param_ranges.keys())
        self.independent_var_names = list(independent_vars_sampling.keys())

        # Check the function signature to understand parameter order
        self._check_function_signature()

    def _check_function_signature(self) -> None:
        """Check the function signature to determine parameter order and names."""
        sig = inspect.signature(self.function)
        params = list(sig.parameters.keys())

        # Verify all independent variables are in the function signature
        for var_name in self.independent_var_names:
            if var_name not in params:
                raise ValueError(f"Independent variable '{var_name}' not found in function signature")

        # Identify the parameter names in the function signature
        self.function_param_indices = {}
        for i, param in enumerate(params):
            if param in self.param_names:
                self.function_param_indices[param] = i

        # Verify all parameters are in the function signature
        missing_params = set(self.param_names) - set(self.function_param_indices.keys())
        if missing_params:
            raise ValueError(f"Parameters {missing_params} not found in function signature")

    def generate_random_parameters(self, n_samples: int) -> np.ndarray:
        """Generate random parameter sets within the specified ranges.

        Args:
            n_samples: Number of parameter sets to generate

        Returns:
            Array of shape (n_samples, n_parameters) containing random parameter values
        """
        params = np.zeros((n_samples, len(self.param_names)))

        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[param_name]
            params[:, i] = np.random.uniform(min_val, max_val, size=n_samples)

        return params

    def evaluate_function(self, params: np.ndarray) -> Dict[str, np.ndarray]:
        """Evaluate the function for each set of parameters.

        Args:
            params: Array of shape (n_samples, n_parameters) containing parameter values

        Returns:
            Dictionary mapping combinations of independent variable values to arrays of
            function values of shape (n_samples, n_points)
        """
        n_samples = params.shape[0]
        results = {}

        # For each combination of independent variable values
        # Simple case: one independent variable
        if len(self.independent_var_names) == 1:
            var_name = self.independent_var_names[0]
            x_values = self.independent_vars_sampling[var_name]
            y_values = np.zeros((n_samples, len(x_values)))

            for i in range(n_samples):
                param_dict = {name: params[i, j] for j, name in enumerate(self.param_names)}
                y_values[i] = self.function(x_values, **param_dict)

            results[var_name] = (x_values, y_values)

        # Complex case: multiple independent variables
        else:
            raise NotImplementedError("Multiple independent variables not yet implemented")

        return results

    def generate_dataset(
        self, n_samples: int, add_noise: bool = False, noise_level: float = 0.05
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Generate a complete dataset of parameters and function values.

        Args:
            n_samples: Number of samples to generate
            add_noise: Whether to add random noise to the function values
            noise_level: Standard deviation of the noise relative to the signal range

        Returns:
            Tuple containing:
            - Array of parameter values of shape (n_samples, n_parameters)
            - Dictionary mapping independent variable names to arrays of function values
        """
        params = self.generate_random_parameters(n_samples)
        y_data = self.evaluate_function(params)

        # Add noise if requested
        if add_noise:
            # Calculate the data range to scale the noise appropriately
            data_range = np.max(y_data) - np.min(y_data)

            # Generate the entire noise matrix at once
            noise = np.random.normal(0, noise_level * data_range, size=y_data.shape)

            # Add noise to the data
            y_data = y_data + noise

        return params, y_data
