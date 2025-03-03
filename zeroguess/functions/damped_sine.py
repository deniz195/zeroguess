"""
Damped sine function implementation for curve fitting.
"""

from typing import Dict, Tuple

import numpy as np

from zeroguess.functions.base import FittingFunction
from zeroguess.functions.constants import DEFAULT_N_INDEPENDENT_POINTS


class DampedSineFunction(FittingFunction):
    """Damped sine function implementation.

    A damped sine function is an oscillatory function with exponential decay.
    It is defined by four parameters: amplitude, frequency, phase, and decay.

    The function has the form: f(x) = amplitude * sin(2π * frequency * x + phase) * exp(-decay * x)
    """

    @property
    def name(self) -> str:
        """Return the name of the function."""
        return "damped_sine"

    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges."""
        return {
            "amplitude": (0.5, 5.0),
            "frequency": (0.5, 3.0),
            "phase": (0.0, 2.0 * np.pi),
            "decay": (0.1, 1.0),
        }

    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            "amplitude": "Initial amplitude of the oscillation",
            "frequency": "Frequency of the oscillation (cycles per unit x)",
            "phase": "Phase offset of the oscillation (radians)",
            "decay": "Exponential decay rate of the amplitude",
        }

    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {"x": np.linspace(0.0, 10.0, DEFAULT_N_INDEPENDENT_POINTS)}

    def __call__(self, x, amplitude, frequency, phase, decay):
        """Evaluate the damped sine function.

        Args:
            x: Independent variable values
            amplitude: Initial amplitude of the oscillation
            frequency: Frequency of the oscillation (cycles per unit x)
            phase: Phase offset of the oscillation (radians)
            decay: Exponential decay rate of the amplitude

        Returns:
            Function values at the specified points
        """
        return amplitude * np.sin(2 * np.pi * frequency * x + phase) * np.exp(-decay * x)
