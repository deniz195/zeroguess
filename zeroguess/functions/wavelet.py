"""
Wavelet function implementation for curve fitting.
"""

from typing import Dict, Tuple

import numpy as np

from zeroguess.functions.base import FittingFunction


class WaveletFunction(FittingFunction):
    """Wavelet function implementation (a modulated Gaussian).

    A wavelet function is a wave-like oscillation with an amplitude that begins at zero,
    increases, and then decreases back to zero. It combines a sine wave with a Gaussian envelope.

    The function has the form: f(x) = envelope * wave
    where:
    - envelope = exp(-((x - position)^2) / (2 * width^2))
    - wave = sin(2π * frequency * x + phase)
    """

    @property
    def name(self) -> str:
        """Return the name of the function."""
        return "wavelet"

    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges."""
        return {
            "frequency": (0.05, 1.0),
            "phase": (0.0, 2.0 * np.pi),
            "position": (5.0, 15.0),
            "width": (0.1, 3.0),
        }

    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            "frequency": "Frequency of the oscillation (cycles per unit x)",
            "phase": "Phase offset of the oscillation (radians)",
            "position": "Center position of the Gaussian envelope",
            "width": "Width of the Gaussian envelope (standard deviation)",
        }

    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {"x": np.linspace(0.0, 20.0, 200)}

    def __call__(self, x, frequency, phase, position, width):
        """Evaluate the wavelet function.

        Args:
            x: Independent variable values
            frequency: Frequency of the oscillation
            phase: Phase offset of the oscillation
            position: Center position of the Gaussian envelope
            width: Width of the Gaussian envelope

        Returns:
            Function values at the specified points
        """
        envelope = np.exp(-((x - position) ** 2) / (2 * width**2))
        wave = np.sin(2 * np.pi * frequency * (x - position) + phase)
        return envelope * wave
