"""
Double sigmoid function implementation for curve fitting.
"""

from typing import Dict, Tuple

import numpy as np

from zeroguess.functions.base import FittingFunction
from zeroguess.functions.constants import DEFAULT_N_INDEPENDENT_POINTS


class DoubleSigmoidFunction(FittingFunction):
    """Double sigmoid function implementation.

    A double sigmoid function is a sum of two sigmoid functions, each with its own
    amplitude, center, and rate parameters. This can model more complex transitions
    or data with two distinct sigmoid-like features.

    The function has the form:
    f(x) = amp1 / (1 + exp(-rate1 * (x - center1))) + amp2 / (1 + exp(-rate2 * (x - center2)))
    """

    @property
    def name(self) -> str:
        """Return the name of the function."""
        return "double_sigmoid"

    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges."""
        return {
            "amp1": (0.3, 5.0),
            "center1": (-5.0, 0.0),
            "rate1": (0.4, 3.0),
            "amp2": (0.3, 5.0),
            "center2": (0.0, 5.0),
            "rate2": (0.4, 3.0),
        }

    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            "amp1": "Maximum value of the first sigmoid curve",
            "center1": "Position of the midpoint of the first curve",
            "rate1": "Steepness of the first curve (growth rate)",
            "amp2": "Maximum value of the second sigmoid curve",
            "center2": "Position of the midpoint of the second curve",
            "rate2": "Steepness of the second curve (growth rate)",
        }

    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {"x": np.linspace(-10.0, 10.0, DEFAULT_N_INDEPENDENT_POINTS)}

    def __call__(self, x, amp1, center1, rate1, amp2, center2, rate2):
        """Evaluate the double sigmoid function.

        Args:
            x: Independent variable values
            amp1: Maximum value of the first sigmoid curve
            center1: Position of the midpoint of the first curve
            rate1: Steepness of the first curve (growth rate)
            amp2: Maximum value of the second sigmoid curve
            center2: Position of the midpoint of the second curve
            rate2: Steepness of the second curve (growth rate)

        Returns:
            Function values at the specified points
        """
        sigmoid1 = amp1 / (1 + np.exp(-rate1 * (x - center1)))
        sigmoid2 = amp2 / (1 + np.exp(-rate2 * (x - center2)))
        return sigmoid1 + sigmoid2
