"""
Neural network architectures for ZeroGuess.
"""

from zeroguess.estimators.architectures.base import BaseArchitecture
from zeroguess.estimators.architectures.cnn import CNNArchitecture
from zeroguess.estimators.architectures.mlp import MLPArchitecture
from zeroguess.estimators.architectures.registry import (
    get_architecture,
    get_architecture_info,
    list_architectures,
    register_architecture,
)
from zeroguess.estimators.architectures.transformer import TransformerArchitecture

__all__ = [
    "BaseArchitecture",
    "MLPArchitecture",
    "CNNArchitecture",
    "TransformerArchitecture",
    "get_architecture",
    "register_architecture",
    "list_architectures",
    "get_architecture_info",
]
