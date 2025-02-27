"""
Registry for neural network architectures.
"""

from typing import Any, Dict, List, Optional, Type

from zeroguess.estimators.architectures.base import BaseArchitecture

# Global registry of architectures
_ARCHITECTURE_REGISTRY: Dict[str, Type[BaseArchitecture]] = {}


def register_architecture(architecture_class: Type[BaseArchitecture]) -> None:
    """Register a neural network architecture with the registry.

    Args:
        architecture_class: Architecture class to register

    Raises:
        ValueError: If architecture with the same name is already registered
    """
    architecture_name = architecture_class.get_name()
    if architecture_name in _ARCHITECTURE_REGISTRY:
        raise ValueError(f"Architecture '{architecture_name}' is already registered")

    _ARCHITECTURE_REGISTRY[architecture_name] = architecture_class


def get_architecture(architecture_name: str, **params) -> BaseArchitecture:
    """Get an instance of the specified architecture.

    Args:
        architecture_name: Name of the architecture to retrieve
        **params: Architecture-specific parameters

    Returns:
        Instance of the requested architecture

    Raises:
        ValueError: If the requested architecture is not registered
    """
    if architecture_name == "default" or architecture_name == "best":
        # Use the default architecture (MLP for now)
        architecture_name = "mlp"
        # architecture_name = "cnn"

    if architecture_name not in _ARCHITECTURE_REGISTRY:
        registered = ", ".join(_ARCHITECTURE_REGISTRY.keys())
        raise ValueError(
            f"Architecture '{architecture_name}' is not registered. "
            f"Available architectures: {registered}"
        )

    architecture_class = _ARCHITECTURE_REGISTRY[architecture_name]
    return architecture_class(**params)


def list_architectures() -> List[str]:
    """List all registered architecture names.

    Returns:
        List of registered architecture names
    """
    return list(_ARCHITECTURE_REGISTRY.keys())


def get_architecture_info() -> Dict[str, Dict[str, Any]]:
    """Get information about all registered architectures.

    Returns:
        Dictionary mapping architecture names to information dictionaries
    """
    info = {}
    for name, cls in _ARCHITECTURE_REGISTRY.items():
        info[name] = {
            "description": cls.get_description(),
            "default_params": cls.get_default_params(),
        }
    return info


from zeroguess.estimators.architectures.cnn import CNNArchitecture

# Register built-in architectures
from zeroguess.estimators.architectures.mlp import MLPArchitecture
from zeroguess.estimators.architectures.transformer import TransformerArchitecture

register_architecture(MLPArchitecture)
# CNNArchitecture and TransformerArchitecture are planned for future work
register_architecture(CNNArchitecture)
# register_architecture(TransformerArchitecture)
