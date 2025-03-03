"""
ZeroGuess: Machine Learning for Curve Fitting Parameter Estimation
=============================================================

ZeroGuess is a Python library that simplifies the estimation of starting parameters
for curve fitting by leveraging machine learning.
"""

__version__ = "0.7.2"

from zeroguess.estimators.base import BaseEstimator

# Import core components to make them available at the top level
from zeroguess.estimators.factory import create_estimator

# Make key classes and functions available at the top level
__all__ = ["create_estimator", "BaseEstimator"]
