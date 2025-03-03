"""
Multimodal function implementation for curve fitting.
"""

from typing import Dict, Tuple

import numpy as np

from zeroguess.functions.base import FittingFunction
from zeroguess.functions.constants import DEFAULT_N_INDEPENDENT_POINTS


class MultimodalFunction(FittingFunction):
    """Multimodal function implementation.

    A multimodal function with local minima, defined as a combination of sine and cosine waves.
    This function creates a complex landscape with multiple local minima, making it useful
    for testing optimization algorithms.

    The function has the form: f(x) = a1 * sin(a2 * x) + a3 * cos(a4 * x + a5)
    """

    @property
    def name(self) -> str:
        """Return the name of the function."""
        return "multimodal"

    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges."""
        return {
            "a1": (0.0, 5.0),  # Amplitude of sine component
            "a2": (0.1, 3.0),  # Frequency of sine component
            "a3": (0.0, 5.0),  # Amplitude of cosine component
            "a4": (0.1, 3.0),  # Frequency of cosine component
            "a5": (0.0, 2.0 * np.pi),  # Phase shift of cosine component
        }

    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            "a1": "Amplitude of sine component",
            "a2": "Frequency of sine component",
            "a3": "Amplitude of cosine component",
            "a4": "Frequency of cosine component",
            "a5": "Phase shift of cosine component",
        }

    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {"x": np.linspace(-10.0, 10.0, DEFAULT_N_INDEPENDENT_POINTS)}

    def __call__(self, x, a1, a2, a3, a4, a5):
        """Evaluate the multimodal function.

        Args:
            x: Independent variable values
            a1: Amplitude of sine component
            a2: Frequency of sine component
            a3: Amplitude of cosine component
            a4: Frequency of cosine component
            a5: Phase shift of cosine component

        Returns:
            Function values at the specified points
        """
        return a1 * np.sin(a2 * x) + a3 * np.cos(a4 * x + a5)
