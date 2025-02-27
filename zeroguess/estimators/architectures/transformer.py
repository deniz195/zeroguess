"""
Transformer architecture for ZeroGuess (Future Work).

Note: This architecture is currently not implemented and is planned for future work.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from zeroguess.estimators.architectures.base import BaseArchitecture


class TransformerArchitecture(BaseArchitecture):
    """Transformer architecture implementation (Future Work)."""

    def __init__(self, **params):
        """Initialize the Transformer architecture with specific parameters.

        Args:
            **params: Architecture-specific parameters
        """
        raise NotImplementedError(
            "Transformer architecture is not yet implemented. "
            "This is planned for future work."
        )

    def create_network(self, n_input_features: int, n_output_params: int) -> nn.Module:
        """Create a Transformer network with the specified input and output dimensions.

        Args:
            n_input_features: Number of input features
            n_output_params: Number of output parameters

        Returns:
            A Transformer module
        """
        raise NotImplementedError(
            "Transformer architecture is not yet implemented. "
            "This is planned for future work."
        )

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Get the default parameters for the Transformer architecture.

        Returns:
            Dictionary of default parameter values
        """
        return {
            "n_heads": 4,
            "n_layers": 2,
            "dim_feedforward": 128,
            "dropout": 0.1,
        }

    @classmethod
    def get_description(cls) -> str:
        """Get a description of the Transformer architecture.

        Returns:
            String description of the architecture
        """
        return "Transformer: Architecture using self-attention mechanisms to capture global relationships (Future Work)."
