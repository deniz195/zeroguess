"""
Standard fitting functions for curve fitting applications.

This module provides implementations of common fitting functions used in
curve fitting applications, such as Gaussian, multi-peak Gaussian, damped sine,
and linear functions.
"""

from typing import Dict, Tuple

import numpy as np

from zeroguess.functions.base import FittingFunction

DEFAULT_N_INDEPENDENT_POINTS = 100


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


class MultiPeakGaussianFunction(FittingFunction):
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
        return "Two-Peak Gaussian Function"

    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges."""
        return {
            "amplitude_1": (0.1, 10.0),
            "center_1": (-5.0, 5.0),
            "width_1": (0.1, 2.0),
            "amplitude_2": (0.1, 10.0),
            "center_2": (-5.0, 5.0),
            "width_2": (0.1, 2.0),
        }

    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            "amplitude_1": "Peak height of the first Gaussian peak",
            "center_1": "Position of the center of the first peak",
            "width_1": "Width of the first peak (standard deviation)",
            "amplitude_2": "Peak height of the second Gaussian peak",
            "center_2": "Position of the center of the second peak",
            "width_2": "Width of the second peak (standard deviation)",
        }

    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {"x": np.linspace(-10.0, 10.0, DEFAULT_N_INDEPENDENT_POINTS)}

    def __call__(self, x, amplitude_1, center_1, width_1, amplitude_2, center_2, width_2):
        """Evaluate the two-peak Gaussian function.

        Args:
            x: Independent variable values
            amplitude_1: Peak height of the first Gaussian peak
            center_1: Position of the center of the first peak
            width_1: Width of the first peak (standard deviation)
            amplitude_2: Peak height of the second Gaussian peak
            center_2: Position of the center of the second peak
            width_2: Width of the second peak (standard deviation)

        Returns:
            Function values at the specified points
        """
        # Calculate the first Gaussian peak
        peak_1 = amplitude_1 * np.exp(-((x - center_1) ** 2) / (2 * width_1**2))

        # Calculate the second Gaussian peak
        peak_2 = amplitude_2 * np.exp(-((x - center_2) ** 2) / (2 * width_2**2))

        # Sum the peaks
        return peak_1 + peak_2


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
            "amp1": (0.0, 5.0),
            "center1": (-5.0, 0.0),
            "rate1": (0.1, 3.0),
            "amp2": (0.0, 5.0),
            "center2": (0.0, 5.0),
            "rate2": (0.1, 3.0),
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
