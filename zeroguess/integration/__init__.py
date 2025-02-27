"""
Integration adapters for curve fitting libraries.
"""

# Create aliases for easier imports
from zeroguess.integration import lmfit_adapter as lmfit_integration
from zeroguess.integration import scipy_adapter as scipy_integration

# Make modules available at the top level
__all__ = ["scipy_integration", "lmfit_integration"]
