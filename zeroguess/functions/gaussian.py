"""
Gaussian function implementation for curve fitting.
"""

from typing import Dict, Tuple

import numpy as np

from zeroguess.functions.base import FittingFunction
from zeroguess.functions.constants import DEFAULT_N_INDEPENDENT_POINTS


class GaussianFunction(FittingFunction):
    """Gaussian function implementation.

    A Gaussian function is a bell-shaped curve defined by three parameters:
    amplitude, center, and width.

    The function has the form: f(x) = amplitude * exp(-(x - center)^2 / (2 * width^2))
    """

    @property
    def name(self) -> str:
        """Return the name of the function."""
        return "gaussian"

    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges."""
        return {"amplitude": (0.5, 10.0), "center": (-5.0, 5.0), "width": (0.1, 2.0)}

    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            "amplitude": "Peak height of the Gaussian curve",
            "center": "Position of the center of the peak",
            "width": "Width of the peak (standard deviation)",
        }

    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {"x": np.linspace(-10.0, 10.0, DEFAULT_N_INDEPENDENT_POINTS)}

    def __call__(self, x, amplitude, center, width):
        """Evaluate the Gaussian function.

        Args:
            x: Independent variable values
            amplitude: Peak height of the Gaussian curve
            center: Position of the center of the peak
            width: Width of the peak (standard deviation)

        Returns:
            Function values at the specified points
        """
        return amplitude * np.exp(-((x - center) ** 2) / (2 * width**2))
