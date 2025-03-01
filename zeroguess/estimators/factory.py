"""
Factory for creating parameter estimator instances.
"""

from typing import Any, Callable, Dict, Tuple

import numpy as np

from zeroguess.estimators.base import BaseEstimator


def create_estimator(
    function: Callable,
    param_ranges: Dict[str, Tuple[float, float]],
    independent_vars_sampling: Dict[str, np.ndarray],
    estimator_type: str = "neural_network",
    architecture: str = "best",
    architecture_params: Dict[str, Any] = None,
    **kwargs,
) -> BaseEstimator:
    """Create a parameter estimator instance.

    Args:
        function: The curve fitting target function
        param_ranges: Dictionary mapping parameter names to (min, max) tuples
        independent_vars_sampling: Dictionary mapping independent variable names
            to arrays of sampling points
        estimator_type: Type of estimator to create
            Options: "neural_network" (default), "nnae" (autoencoder)
        architecture: Neural network architecture to use (default: "best")
                     Available options depend on the estimator type
        architecture_params: Architecture-specific parameters
        **kwargs: Additional arguments to pass to the estimator constructor

    Returns:
        Instance of a BaseEstimator subclass

    Raises:
        ValueError: If the estimator type or architecture is not recognized
    """
    if estimator_type == "neural_network":
        # Import here to avoid circular imports
        from zeroguess.estimators.nn_estimator import NeuralNetworkEstimator

        # Handle architecture selection
        if architecture_params is None:
            architecture_params = {}

        return NeuralNetworkEstimator(
            function=function,
            param_ranges=param_ranges,
            independent_vars_sampling=independent_vars_sampling,
            architecture=architecture,
            architecture_params=architecture_params,
            **kwargs,
        )
    elif estimator_type == "nnae":
        # Import here to avoid circular imports
        from zeroguess.estimators.nnae_estimator import NNAEEstimator

        # Handle architecture selection
        if architecture_params is None:
            architecture_params = {}

        return NNAEEstimator(
            function=function,
            param_ranges=param_ranges,
            independent_vars_sampling=independent_vars_sampling,
            architecture=architecture,
            architecture_params=architecture_params,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown estimator type: {estimator_type}")


# Export the create_estimator function to the top-level package
__all__ = ["create_estimator"]
