"""
Multilayer Perceptron (MLP) architecture for ZeroGuess.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from zeroguess.estimators.architectures.base import BaseArchitecture


class MLPNetwork(nn.Module):
    """Multilayer Perceptron network for parameter estimation."""

    def __init__(
        self,
        n_input_features: int,
        n_output_params: int,
        hidden_layers: List[int] = [128, 256, 128, 64],
        activation: str = "relu",
        dropout_rate: float = 0.02,
        use_batch_norm: bool = True,
        use_residual: bool = False,
    ):
        """Initialize the MLP network.

        Args:
            n_input_features: Number of input features
            n_output_params: Number of output parameters
            hidden_layers: List of hidden layer sizes
            activation: Activation function name (relu, tanh, sigmoid, etc.)
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
            use_residual: Whether to use residual connections
        """
        super().__init__()

        # Select activation function
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "sigmoid":
            act_fn = nn.Sigmoid()
        elif activation == "leaky_relu":
            act_fn = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Create the network layers
        layers = []
        prev_size = n_input_features

        # Add hidden layers
        for i, size in enumerate(hidden_layers):
            # Add linear layer
            layers.append(nn.Linear(prev_size, size))

            # Add activation
            layers.append(act_fn)

            # Add batch normalization if requested
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(size))

            # Add dropout for regularization
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            # Save layer size for residual connections or next layer
            prev_size = size

        # Add output layer
        layers.append(nn.Linear(prev_size, n_output_params))

        # Add sigmoid activation to ensure output is between 0 and 1
        layers.append(nn.Sigmoid())

        # Create the sequential model
        self.model = nn.Sequential(*layers)

        # Store configuration for residual connections
        self.use_residual = use_residual
        self.hidden_layers = hidden_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, n_input_features)

        Returns:
            Output tensor of shape (batch_size, n_output_params)
        """
        if not self.use_residual:
            return self.model(x)

        # For residual network implementation, would go here
        # This is left as a placeholder for future enhancements
        return self.model(x)


class MLPArchitecture(BaseArchitecture):
    """Multilayer Perceptron architecture implementation."""

    def __init__(self, **params):
        """Initialize the MLP architecture with specific parameters.

        Args:
            **params: Architecture-specific parameters
        """
        self.params = self.validate_params(params)

    def create_network(self, n_input_features: int, n_output_params: int) -> nn.Module:
        """Create an MLP network with the specified input and output dimensions.

        Args:
            n_input_features: Number of input features
            n_output_params: Number of output parameters

        Returns:
            An MLPNetwork module
        """
        return MLPNetwork(
            n_input_features=n_input_features,
            n_output_params=n_output_params,
            **self.params,
        )

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Get the default parameters for the MLP architecture.

        Returns:
            Dictionary of default parameter values
        """
        return {
            "hidden_layers": [128, 256, 256, 128, 64],
            "activation": "relu",
            "dropout_rate": 0.02,
            "use_batch_norm": True,
            "use_residual": False,
        }

    @classmethod
    def get_description(cls) -> str:
        """Get a description of the MLP architecture.

        Returns:
            String description of the architecture
        """
        return "Multilayer Perceptron: A standard feedforward neural network with fully connected layers."
