"""
Sigmoid function implementation for curve fitting.
"""

from typing import Dict, Tuple

import numpy as np

from zeroguess.functions.base import FittingFunction
from zeroguess.functions.constants import DEFAULT_N_INDEPENDENT_POINTS


class SigmoidFunction(FittingFunction):
    """Sigmoid/logistic function implementation.

    A sigmoid function is an S-shaped curve defined by three parameters:
    amplitude, center, and rate.

    The function has the form: f(x) = amplitude / (1 + exp(-rate * (x - center)))
    """

    @property
    def name(self) -> str:
        """Return the name of the function."""
        return "sigmoid"

    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges."""
        return {"amplitude": (0.0, 10.0), "center": (-5.0, 5.0), "rate": (0.1, 5.0)}

    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            "amplitude": "Maximum value of the sigmoid curve",
            "center": "Position of the midpoint of the curve",
            "rate": "Steepness of the curve (growth rate)",
        }

    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {"x": np.linspace(-10.0, 10.0, DEFAULT_N_INDEPENDENT_POINTS)}

    def __call__(self, x, amplitude, center, rate):
        """Evaluate the sigmoid function.

        Args:
            x: Independent variable values
            amplitude: Maximum value of the sigmoid curve
            center: Position of the midpoint of the curve
            rate: Steepness of the curve (growth rate)

        Returns:
            Function values at the specified points
        """
        return amplitude / (1 + np.exp(-rate * (x - center)))
