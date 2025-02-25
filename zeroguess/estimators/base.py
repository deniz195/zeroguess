"""
Base interface for parameter estimators.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union, Callable

import numpy as np


class BaseEstimator(ABC):
    """Base class for all parameter estimators in ZeroGuess.
    
    This abstract class defines the interface that all parameter estimators
    must implement. It provides the basic structure for training and prediction.
    """
    
    def __init__(
        self,
        function: Callable,
        param_ranges: Dict[str, Tuple[float, float]],
        independent_vars_sampling: Dict[str, np.ndarray],
        **kwargs
    ):
        """Initialize the estimator.
        
        Args:
            function: The curve fitting target function
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            independent_vars_sampling: Dictionary mapping independent variable names
                to arrays of sampling points
            **kwargs: Additional keyword arguments for specific estimator implementations
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
        """Validate the inputs provided to the estimator."""
        if not callable(self.function):
            raise TypeError("Function must be callable")
        
        if not isinstance(self.param_ranges, dict) or not self.param_ranges:
            raise ValueError("param_ranges must be a non-empty dictionary")
        
        for param_name, param_range in self.param_ranges.items():
            if not isinstance(param_range, tuple) or len(param_range) != 2:
                raise ValueError(f"Range for parameter {param_name} must be a tuple of (min, max)")
            if param_range[0] >= param_range[1]:
                raise ValueError(f"Min value must be less than max value for parameter {param_name}")
        
        if not isinstance(self.independent_vars_sampling, dict) or not self.independent_vars_sampling:
            raise ValueError("independent_vars_sampling must be a non-empty dictionary")
        
        for var_name, sampling in self.independent_vars_sampling.items():
            if not isinstance(sampling, np.ndarray):
                raise ValueError(f"Sampling points for {var_name} must be a numpy array")
            if sampling.size == 0:
                raise ValueError(f"Sampling points for {var_name} must not be empty")
    
    @abstractmethod
    def train(self, n_samples: int = 1000, **kwargs) -> Dict[str, Any]:
        """Train the estimator on synthetic data.
        
        Args:
            n_samples: Number of synthetic data samples to generate
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics and information
        """
        pass
    
    @abstractmethod
    def predict(self, *args, **kwargs) -> Dict[str, float]:
        """Predict parameter values for the given data.
        
        The specific signature depends on the implementation, but typically:
        - For single independent variable: predict(x_data, y_data)
        - For multiple independent variables: predict(y_data, x1=x1_data, x2=x2_data, ...)
        
        Returns:
            Dictionary mapping parameter names to estimated values
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save the trained estimator to a file.
        
        Args:
            path: Path to save the estimator to
        """
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseEstimator':
        """Load a trained estimator from a file.
        
        Args:
            path: Path to load the estimator from
            
        Returns:
            Loaded estimator instance
        """
        pass
