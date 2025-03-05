"""
Integration adapters for curve fitting libraries.
"""

from zeroguess.integration import scipy_adapter as scipy_integration

from .lmfit_adapter import Model as ZeroGuessModel

# Make modules available at the top level
__all__ = ["scipy_integration", "ZeroGuessModel"]
