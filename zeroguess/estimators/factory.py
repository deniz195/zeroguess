"""
Factory for creating parameter estimator instances.
"""
from typing import Any, Callable, Dict, Optional, Tuple, Type, Union

import numpy as np

from zeroguess.estimators.base import BaseEstimator


def create_estimator(
    function: Callable,
    param_ranges: Dict[str, Tuple[float, float]],
    independent_vars_sampling: Dict[str, np.ndarray],
    estimator_type: str = "neural_network",
    **kwargs
) -> BaseEstimator:
    """Create a parameter estimator instance.
    
    Args:
        function: The curve fitting target function
        param_ranges: Dictionary mapping parameter names to (min, max) tuples
        independent_vars_sampling: Dictionary mapping independent variable names
            to arrays of sampling points
        estimator_type: Type of estimator to create (default: "neural_network")
        **kwargs: Additional arguments to pass to the estimator constructor
        
    Returns:
        Instance of a BaseEstimator subclass
        
    Raises:
        ValueError: If the estimator type is not recognized
    """
    if estimator_type == "neural_network":
        # Import here to avoid circular imports
        from zeroguess.estimators.nn_estimator import NeuralNetworkEstimator
        return NeuralNetworkEstimator(
            function=function,
            param_ranges=param_ranges,
            independent_vars_sampling=independent_vars_sampling,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")


# Export the create_estimator function to the top-level package
__all__ = ["create_estimator"]
