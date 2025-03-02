"""
Integration with SciPy's curve_fit function.
"""

import inspect
from typing import Callable, Dict, Optional, Tuple

import numpy as np
from scipy import optimize

import zeroguess


def curve_fit(
    f: Callable,
    xdata: np.ndarray,
    ydata: np.ndarray,
    param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
    independent_vars_sampling: Optional[Dict[str, np.ndarray]] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Enhanced version of scipy.optimize.curve_fit with automatic parameter estimation.

    This function wraps scipy.optimize.curve_fit and adds automatic initial parameter
    estimation using ZeroGuess.

    Args:
        f: The fitting function with signature f(x, *params) or f(x, param1, param2, ...)
        xdata: Independent variable data points
        ydata: Dependent variable data points
        param_ranges: Dictionary mapping parameter names to (min, max) tuples
        independent_vars_sampling: Dictionary mapping independent variable names
            to arrays of sampling points for training
        **kwargs: Additional keyword arguments to pass to scipy.optimize.curve_fit

    Returns:
        Tuple containing:
        - Array of optimal parameter values
        - Covariance matrix
    """
    # If no parameter ranges provided, fall back to standard curve_fit
    if param_ranges is None:
        return optimize.curve_fit(f, xdata, ydata, **kwargs)

    # Get function signature to determine parameter names
    sig = inspect.signature(f)
    param_names = list(sig.parameters.keys())[1:]  # Skip first (x) parameter

    # Ensure all parameters have ranges
    for param_name in param_names:
        if param_name not in param_ranges:
            raise ValueError(f"No range provided for parameter '{param_name}'")

    # Set up independent variable sampling if not provided
    if independent_vars_sampling is None:
        # Default to using the provided xdata as sampling points
        independent_vars_sampling = {"x": xdata}

    # Create estimator
    estimator = zeroguess.create_estimator(
        function=f,
        param_ranges=param_ranges,
        independent_vars_sampling=independent_vars_sampling,
    )

    # Train the estimator
    estimator.train()

    # Get parameter estimates
    p0_dict = estimator.predict(xdata, ydata)

    # Convert dictionary to array, keeping the same order as in the function signature
    p0 = np.array([p0_dict[param_name] for param_name in param_names])

    # If p0 is already provided in kwargs, warn and use it instead
    if "p0" in kwargs:
        import warnings

        warnings.warn(
            "Initial parameter guess (p0) already provided. " "ZeroGuess's parameter estimation will be ignored.",
            stacklevel=2,
        )
    else:
        kwargs["p0"] = p0

    # Call the original curve_fit function
    return optimize.curve_fit(f, xdata, ydata, **kwargs)
