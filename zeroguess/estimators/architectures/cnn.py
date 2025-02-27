"""
CNN architecture for parameter estimation in ZeroGuess.

This module implements a 1D Convolutional Neural Network architecture
optimized for curve fitting problems, especially those with oscillatory
or multi-peak characteristics.
"""

from typing import Any, Dict, List

import torch
import torch.nn as nn

from zeroguess.estimators.architectures.base import BaseArchitecture


class CNNNetwork(nn.Module):
    """1D Convolutional Neural Network for parameter estimation."""

    def __init__(
        self,
        n_input_features: int,
        n_output_params: int,
        n_conv_layers: int = 3,
        filters: List[int] = [16, 32, 64],
        kernel_size: int = 5,
        pool_size: int = 2,
        fc_units: List[int] = [128, 64],
        activation: str = "relu",
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
    ):
        """Initialize the CNN network.

        Args:
            n_input_features: Number of input features (data points in the curve)
            n_output_params: Number of output parameters to estimate
            n_conv_layers: Number of convolutional layers
            filters: List of filter counts for each conv layer
            kernel_size: Size of convolutional kernels
            pool_size: Size of pooling windows
            fc_units: List of fully connected layer sizes
            activation: Activation function name (relu, tanh, etc.)
            dropout_rate: Dropout rate for regularization
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()

        # Input validation
        if n_conv_layers != len(filters):
            raise ValueError(
                f"Number of conv layers ({n_conv_layers}) must match length of filters list ({len(filters)})"
            )

        # Select activation function
        if activation == "relu":
            self.act_fn = nn.ReLU()
        elif activation == "tanh":
            self.act_fn = nn.Tanh()
        elif activation == "leaky_relu":
            self.act_fn = nn.LeakyReLU(0.1)
        else:
            raise ValueError(f"Unsupported activation function: {activation}")

        # Build convolutional layers
        self.conv_layers = nn.ModuleList()

        # Reshape input for 1D convolution: [batch_size, 1, n_input_features]
        self.conv_layers.append(nn.Unflatten(1, (1, n_input_features)))

        in_channels = 1  # Start with 1 channel (raw signal)
        feature_length = n_input_features

        for i in range(n_conv_layers):
            # Add convolutional layer
            self.conv_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=filters[i],
                    kernel_size=kernel_size,
                    padding="same",  # Keep spatial dimensions the same
                )
            )

            # Add batch normalization if requested
            if use_batch_norm:
                self.conv_layers.append(nn.BatchNorm1d(filters[i]))

            # Add activation
            self.conv_layers.append(self.act_fn)

            # Add pooling to reduce dimensionality
            self.conv_layers.append(nn.MaxPool1d(pool_size))

            # Update dimensions for next layer
            in_channels = filters[i]
            feature_length = feature_length // pool_size

        # Calculate the flattened size after convolutions and pooling
        self.flattened_size = feature_length * filters[-1]

        # Create fully connected layers
        self.fc_layers = nn.ModuleList()

        in_features = self.flattened_size
        for units in fc_units:
            self.fc_layers.append(nn.Linear(in_features, units))
            if use_batch_norm:
                self.fc_layers.append(nn.BatchNorm1d(units))
            self.fc_layers.append(self.act_fn)
            if dropout_rate > 0:
                self.fc_layers.append(nn.Dropout(dropout_rate))
            in_features = units

        # Output layer
        self.output_layer = nn.Linear(in_features, n_output_params)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape [batch_size, n_input_features]

        Returns:
            Output tensor of shape [batch_size, n_output_params]
        """
        # Process through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten the output of convolutional part
        x = x.view(x.size(0), -1)

        # Process through fully connected layers
        for layer in self.fc_layers:
            x = layer(x)

        # Output layer (parameter prediction)
        x = self.output_layer(x)

        return x


class CNNArchitecture(BaseArchitecture):
    """CNN architecture implementation."""

    def __init__(self, **params):
        """Initialize the CNN architecture with specific parameters.

        Args:
            **params: Architecture-specific parameters
        """
        self.params = self.validate_params(params)

    def create_network(self, n_input_features: int, n_output_params: int) -> nn.Module:
        """Create a CNN network with the specified input and output dimensions.

        Args:
            n_input_features: Number of input features
            n_output_params: Number of output parameters

        Returns:
            A CNNNetwork module
        """
        return CNNNetwork(
            n_input_features=n_input_features,
            n_output_params=n_output_params,
            **self.params,
        )

    def validate_input_size(self, network: nn.Module, input_size: int, expected_size: int) -> bool:
        """Validate if the input size is compatible with the CNN network.

        For CNN architecture, we need to check for the Unflatten layer which reshapes the input.
        We need to validate against the number of features expected by the network, not the
        channel dimension that appears after reshaping.

        Args:
            network: The neural network model (CNN)
            input_size: The size of the input data
            expected_size: The expected input size based on network's first layer

        Returns:
            True if the input size is valid, False otherwise

        Raises:
            ValueError: If input size is incompatible with the network
        """
        # Check if the network has the expected structure for a CNN
        if hasattr(network, "conv_layers") and len(network.conv_layers) > 0:
            # For CNN, check if the first layer is an Unflatten layer
            if isinstance(network.conv_layers[0], nn.Unflatten):
                # Extract the expected feature dimension from the Unflatten layer
                # The Unflatten layer's unflattened_size contains (channels, features)
                _, expected_features = network.conv_layers[0].unflattened_size

                # Compare input size with expected features
                if input_size != expected_features:
                    raise ValueError(
                        f"Input data size ({input_size}) does not match the CNN's expected input size "
                        f"({expected_features}). The network must be trained with the same number of data points "
                        f"as used for prediction to ensure consistent results."
                    )
                return True

        # If we got here, it means the network structure doesn't match our expectations
        # Fall back to the default validation which will raise an error for mismatched sizes
        return super().validate_input_size(network, input_size, expected_size)

    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Get the default parameters for the CNN architecture.

        Returns:
            Dictionary of default parameter values
        """
        return {
            "n_conv_layers": 3,
            "filters": [16, 32, 64],
            "kernel_size": 5,
            "pool_size": 2,
            "fc_units": [128, 64],
            "activation": "relu",
            "dropout_rate": 0.1,
            "use_batch_norm": True,
        }

    @classmethod
    def get_description(cls) -> str:
        """Get a description of the CNN architecture.

        Returns:
            String description of the architecture
        """
        return (
            "Convolutional Neural Network: Specialized for detecting patterns in curves, "
            "particularly effective for oscillatory or multi-peak functions."
        )
