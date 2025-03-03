"""
Double Gaussian function implementation for curve fitting.
"""

from typing import Dict, Tuple

import numpy as np

from zeroguess.functions.base import FittingFunction
from zeroguess.functions.constants import DEFAULT_N_INDEPENDENT_POINTS


class DoubleGaussianFunction(FittingFunction):
    """Multi-peak Gaussian function implementation.

    A multi-peak Gaussian function is a sum of multiple Gaussian functions,
    each with its own amplitude, center, and width parameters.

    This implementation supports exactly 2 Gaussian peaks.
    """

    def __init__(self):
        """Initialize the multi-peak Gaussian function with 2 peaks."""
        super().__init__()

    @property
    def name(self) -> str:
        """Return the name of the function."""
        return "double_gaussian"

    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges."""
        return {
            "amplitude1": (0.1, 10.0),
            "center1": (-5.0, 5.0),
            "width1": (0.1, 2.0),
            "amplitude2": (0.1, 10.0),
            "center2": (-5.0, 5.0),
            "width2": (0.1, 2.0),
        }

    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            "amplitude1": "Peak height of the first Gaussian peak",
            "center1": "Position of the center of the first peak",
            "width1": "Width of the first peak (standard deviation)",
            "amplitude2": "Peak height of the second Gaussian peak",
            "center2": "Position of the center of the second peak",
            "width2": "Width of the second peak (standard deviation)",
        }

    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {"x": np.linspace(-10.0, 10.0, DEFAULT_N_INDEPENDENT_POINTS)}

    def get_canonical_params(self, params: Dict[str, float]) -> Dict[str, float]:
        """Get the canonical parameters for the function.

        Returns:
            Dictionary of canonical parameter values.
        """
        # Sort the parameters by center value
        if params["center1"] > params["center2"]:
            params = {
                "amplitude1": params["amplitude2"],
                "center1": params["center2"],
                "width1": params["width2"],
                "amplitude2": params["amplitude1"],
                "center2": params["center1"],
                "width2": params["width1"],
            }

        return params

    def __call__(self, x, amplitude1, center1, width1, amplitude2, center2, width2):
        """Evaluate the two-peak Gaussian function.

        Args:
            x: Independent variable values
            amplitude1: Peak height of the first Gaussian peak
            center1: Position of the center of the first peak
            width1: Width of the first peak (standard deviation)
            amplitude2: Peak height of the second Gaussian peak
            center2: Position of the center of the second peak
            width2: Width of the second peak (standard deviation)

        Returns:
            Function values at the specified points
        """
        # Calculate the first Gaussian peak
        peak1 = amplitude1 * np.exp(-((x - center1) ** 2) / (2 * width1**2))

        # Calculate the second Gaussian peak
        peak2 = amplitude2 * np.exp(-((x - center2) ** 2) / (2 * width2**2))

        # Sum the peaks
        return peak1 + peak2
