"""
Functions module for ZeroGuess.

This module provides implementations of common fitting functions
used in curve fitting applications.
"""

from zeroguess.functions.base import FittingFunction
from zeroguess.functions.damped_sine import DampedSineFunction
from zeroguess.functions.double_gaussian import DoubleGaussianFunction
from zeroguess.functions.double_sigmoid import DoubleSigmoidFunction
from zeroguess.functions.gaussian import GaussianFunction
from zeroguess.functions.linear import LinearFunction
from zeroguess.functions.multimodal import MultimodalFunction
from zeroguess.functions.sigmoid import SigmoidFunction

# Import utilities
from zeroguess.functions.utils import add_gaussian_noise, signal_to_noise_ratio
from zeroguess.functions.wavelet import WaveletFunction

__all__ = [
    "FittingFunction",
    "GaussianFunction",
    "DoubleGaussianFunction",
    "MultimodalFunction",
    "DampedSineFunction",
    "LinearFunction",
    "SigmoidFunction",
    "DoubleSigmoidFunction",
    "WaveletFunction",
    "add_gaussian_noise",
    "signal_to_noise_ratio",
]
