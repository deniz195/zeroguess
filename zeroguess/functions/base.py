"""
Base class for fitting functions.

This module defines the FittingFunction base class that all fitting function implementations
must inherit from. It provides a common interface for working with curve fitting functions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import numpy as np


class FittingFunction(ABC):
    """Base class for fitting functions.

    This abstract class defines the interface that all fitting function implementations
    must follow. It provides methods for evaluating the function, generating data,
    and accessing metadata about the function such as parameter ranges and descriptions.

    Subclasses must implement all abstract methods and properties.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of the function.

        Returns:
            The function's name as a string.
        """

    @property
    @abstractmethod
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges.

        Returns:
            A dictionary mapping parameter names to (min, max) tuples.
        """

    @property
    @abstractmethod
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls.

        Returns:
            A dictionary mapping parameter names to description strings.
        """

    @property
    @abstractmethod
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables.

        Returns:
            A dictionary mapping independent variable names to numpy arrays of sampling points.
        """

    @abstractmethod
    def __call__(self, *args, **kwargs) -> np.ndarray:
        """Evaluate the function with the given parameters.

        Args:
            *args: Positional arguments to the function.
            **kwargs: Keyword arguments to the function.

        Returns:
            The function values as a numpy array.
        """

    def generate_data(
        self,
        params: Optional[Dict[str, float]] = None,
        indep_vars: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        """Generate data for the function with the given parameters.

        Args:
            params: Dictionary of parameter values. If None, random values within
                the parameter ranges will be used.
            indep_vars: Dictionary of independent variable sampling points. If None,
                the default sampling points will be used.

        Returns:
            A tuple containing:
                - A dictionary of independent variable values
                - A numpy array of dependent variable values
        """
        # Use default parameter ranges if none provided
        if params is None:
            params = {}
            for param_name, (min_val, max_val) in self.param_ranges.items():
                params[param_name] = min_val + (max_val - min_val) * np.random.random()

        # Use default independent variable sampling if none provided
        if indep_vars is None:
            indep_vars = self.default_independent_vars.copy()

        # Evaluate the function with the given parameters
        # Need to handle different possible call signatures based on function implementation
        if len(indep_vars) == 1:
            # Single independent variable case (most common)
            x = list(indep_vars.values())[0]
            x_name = list(indep_vars.keys())[0]
            y = self(x, **params)
            return {x_name: x}, y
        else:
            # Multiple independent variables case
            # This assumes the function implementation can handle a dictionary of independent variables
            y = self(**indep_vars, **params)
            return indep_vars.copy(), y

    @property
    def __name__(self) -> str:
        """Return the name of the function.

        Returns:
            The function's name as a string.
        """
        return self.name

    def __repr__(self) -> str:
        """Return a string representation of the function.

        Returns:
            A string representation of the function.
        """
        return f"{self.__class__.__name__}(name={self.name})"

    def get_canonical_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Get the canonical parameters for the function.

        Returns:
            Dictionary of canonical parameter values.
        """
        return params

    def get_random_params(self, canonical: bool = True) -> Dict[str, float]:
        """Generate random parameters within the defined ranges.

        Returns:
            Dictionary of randomly generated parameter values.
        """
        params = {}
        for param_name, (min_val, max_val) in self.param_ranges.items():
            params[param_name] = min_val + (max_val - min_val) * np.random.random()

        if canonical:
            return self.get_canonical_params(params)
        else:
            return params
