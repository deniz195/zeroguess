"""
Linear function implementation for curve fitting.
"""

from typing import Dict, Tuple

import numpy as np

from zeroguess.functions.base import FittingFunction
from zeroguess.functions.constants import DEFAULT_N_INDEPENDENT_POINTS


class LinearFunction(FittingFunction):
    """Linear function implementation.

    A linear function is a simple straight line defined by two parameters:
    slope and intercept.

    The function has the form: f(x) = slope * x + intercept
    """

    @property
    def name(self) -> str:
        """Return the name of the function."""
        return "linear"

    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges."""
        return {"slope": (-5.0, 5.0), "intercept": (-10.0, 10.0)}

    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            "slope": "Rate of change of the function (rise over run)",
            "intercept": "Value of the function when x = 0 (y-intercept)",
        }

    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {"x": np.linspace(-10.0, 10.0, DEFAULT_N_INDEPENDENT_POINTS)}

    def __call__(self, x, slope, intercept):
        """Evaluate the linear function.

        Args:
            x: Independent variable values
            slope: Rate of change of the function
            intercept: Value of the function when x = 0

        Returns:
            Function values at the specified points
        """
        return slope * x + intercept
