"""
Fitting function collection for ZeroGuess.

This submodule provides a collection of pre-defined fitting functions commonly
used in curve fitting applications, each encapsulated in a class that provides
the function implementation, default parameter ranges, and default independent
variable sampling points.

Each function class also provides parameter descriptions to help users understand
what each parameter controls.
"""

# Import the base class
from zeroguess.functions.base import FittingFunction

# Import standard functions
from zeroguess.functions.standard import (
    GaussianFunction,
    MultiPeakGaussianFunction,
    DampedSineFunction,
    LinearFunction
)

# Import utility functions
from zeroguess.functions.utils import add_gaussian_noise, signal_to_noise_ratio

# Define what's available to import with 'from zeroguess.functions import *'
__all__ = [
    # Base class
    'FittingFunction',
    
    # Standard functions
    'GaussianFunction',
    'MultiPeakGaussianFunction',
    'DampedSineFunction',
    'LinearFunction',
    
    # Utility functions
    'add_gaussian_noise',
    'signal_to_noise_ratio'
] 