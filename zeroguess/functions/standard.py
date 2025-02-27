"""
Standard fitting functions for curve fitting applications.

This module provides implementations of common fitting functions used in
curve fitting applications, such as Gaussian, multi-peak Gaussian, damped sine, 
and linear functions.
"""

import numpy as np
from typing import Dict, Tuple, List, Optional

from zeroguess.functions.base import FittingFunction


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
        return {
            'amplitude': (0.5, 10.0),
            'center': (-5.0, 5.0),
            'width': (0.1, 2.0)
        }
        
    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            'amplitude': 'Peak height of the Gaussian curve',
            'center': 'Position of the center of the peak',
            'width': 'Width of the peak (standard deviation)'
        }
        
    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {
            'x': np.linspace(-10.0, 10.0, 200)
        }
    
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
        return amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))


class MultiPeakGaussianFunction(FittingFunction):
    """Multi-peak Gaussian function implementation.
    
    A multi-peak Gaussian function is a sum of multiple Gaussian functions,
    each with its own amplitude, center, and width parameters.
    
    This implementation supports two peaks by default.
    """
    
    def __init__(self, n_peaks: int = 2):
        """Initialize the multi-peak Gaussian function.
        
        Args:
            n_peaks: Number of Gaussian peaks to include
        """
        self.n_peaks = n_peaks
        super().__init__()
        
    @property
    def name(self) -> str:
        """Return the name of the function."""
        return f"multi_peak_gaussian_{self.n_peaks}"
        
    @property
    def param_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Return the default parameter ranges."""
        ranges = {}
        for i in range(1, self.n_peaks + 1):
            ranges[f'amplitude_{i}'] = (0.5, 10.0)
            ranges[f'center_{i}'] = (-5.0, 5.0)
            ranges[f'width_{i}'] = (0.1, 2.0)
        return ranges
        
    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        descriptions = {}
        for i in range(1, self.n_peaks + 1):
            descriptions[f'amplitude_{i}'] = f'Peak height of Gaussian {i}'
            descriptions[f'center_{i}'] = f'Position of the center of peak {i}'
            descriptions[f'width_{i}'] = f'Width of peak {i} (standard deviation)'
        return descriptions
        
    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {
            'x': np.linspace(-10.0, 10.0, 200)
        }
    
    def __call__(self, x, **params):
        """Evaluate the multi-peak Gaussian function.
        
        Args:
            x: Independent variable values
            **params: Parameters for each Gaussian peak, with names in the format
                amplitude_i, center_i, width_i where i is the peak number (1-based)
                
        Returns:
            Function values at the specified points
        """
        result = np.zeros_like(x, dtype=float)
        
        for i in range(1, self.n_peaks + 1):
            amplitude = params.get(f'amplitude_{i}', 0.0)
            center = params.get(f'center_{i}', 0.0)
            width = params.get(f'width_{i}', 1.0)
            
            # Add the contribution of this peak
            result += amplitude * np.exp(-(x - center) ** 2 / (2 * width ** 2))
            
        return result


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
            'amplitude': (0.5, 5.0),
            'frequency': (0.5, 3.0),
            'phase': (0.0, 2.0 * np.pi),
            'decay': (0.1, 1.0)
        }
        
    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            'amplitude': 'Initial amplitude of the oscillation',
            'frequency': 'Frequency of the oscillation (cycles per unit x)',
            'phase': 'Phase offset of the oscillation (radians)',
            'decay': 'Exponential decay rate of the amplitude'
        }
        
    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {
            'x': np.linspace(0.0, 10.0, 200)
        }
    
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
        return {
            'slope': (-5.0, 5.0),
            'intercept': (-10.0, 10.0)
        }
        
    @property
    def param_descriptions(self) -> Dict[str, str]:
        """Return descriptions of what each parameter controls."""
        return {
            'slope': 'Rate of change of the function (rise over run)',
            'intercept': 'Value of the function when x = 0 (y-intercept)'
        }
        
    @property
    def default_independent_vars(self) -> Dict[str, np.ndarray]:
        """Return default sampling points for independent variables."""
        return {
            'x': np.linspace(-10.0, 10.0, 100)
        }
    
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