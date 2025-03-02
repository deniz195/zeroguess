"""
Neural network autoencoder-based parameter estimator implementation.
"""

import os
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from zeroguess.data.generators import SyntheticDataGenerator
from zeroguess.estimators.base import BaseEstimator


class NNAEEstimator(BaseEstimator):
    """Neural Network Auto-Encoder Estimator for parameter fitting."""

    def __init__(
        self,
        function,
        param_ranges,
        independent_vars=None,
        independent_vars_sampling=None,
        encoder_layers=None,
        decoder_layers=None,
        learning_rate=0.001,
        alpha=0.5,
        beta=0.5,
        architecture="mlp",
        architecture_params=None,
        device=None,
    ):
        """
        Initialize the NNAE estimator.

        Args:
            function: The function to fit parameters to
            param_ranges: Dictionary of parameter ranges {param_name: (min, max)}
            independent_vars: Dictionary of independent variables {var_name: values}
            independent_vars_sampling: Dictionary of independent variables for sampling {var_name: values}
            encoder_layers: List of layer sizes for the encoder (default: [128, 256, 256, 128, 64])
            decoder_layers: List of layer sizes for the decoder (default: [64, 128, 256, 256, 128])
            learning_rate: Learning rate for optimizers
            alpha: Weight for the moment loss component (default: 0.5)
            beta: Weight for the parameter validation loss component (default: 0.5)
            architecture: Name of the architecture to use (default: "mlp")
            architecture_params: Dictionary of architecture parameters
            device: Device to use for computation (default: cuda if available, else cpu)
        """
        self.function = function
        self.param_ranges = param_ranges
        self.independent_vars = independent_vars
        self.param_names = list(param_ranges.keys())
        self.alpha = alpha
        self.beta = beta
        self.architecture = architecture

        # Set up independent variables for sampling if not provided
        if independent_vars_sampling is None:
            self.independent_vars_sampling = independent_vars
        else:
            self.independent_vars_sampling = independent_vars_sampling

        # Store the names of independent variables
        self.independent_var_names = list(self.independent_vars_sampling.keys())

        # Set default encoder and decoder layers if not provided
        if encoder_layers is None:
            encoder_layers = [128, 256, 256, 128, 64]
        if decoder_layers is None:
            decoder_layers = [64, 128, 256, 256, 128]

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        # Set up architecture params
        if architecture_params is None:
            self.architecture_params = {"encoder_layers": self.encoder_layers, "decoder_layers": self.decoder_layers}
        else:
            self.architecture_params = architecture_params

        self.learning_rate = learning_rate

        # Use GPU if available
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # These will be initialized during training
        self.network = None
        self.data_generator = None

        # Initialize the network architecture
        self._create_architecture()

    def _create_architecture(self):
        """Create the network architecture based on architecture parameters."""
        # Determine input dimensions
        input_dim = sum(len(points) for points in self.independent_vars_sampling.values())
        param_dim = len(self.param_ranges)

        # Create network using parameters already set in self.architecture_params
        self.network = _NNAENetwork(
            input_size=input_dim,
            n_params=param_dim,
            encoder_layers=self.encoder_layers,
            decoder_layers=self.decoder_layers,
            n_moments=10,
        ).to(self.device)

    @staticmethod
    def list_available_architectures() -> List[str]:
        """List all available neural network architectures for NNAE.

        Returns:
            List of available architecture names
        """
        return ["mlp", "default", "best"]

    @staticmethod
    def get_architecture_details() -> Dict[str, Dict[str, Any]]:
        """Get details about all available architectures for NNAE.

        Returns:
            Dictionary mapping architecture names to information dictionaries
            containing description and default parameters
        """
        return {
            "mlp": {
                "description": "Multilayer Perceptron with encoder-decoder structure",
                "default_params": {
                    "encoder_layers": [128, 256, 256, 128, 64],
                    "decoder_layers": [64, 128, 256, 256, 128],
                    "activation": "relu",
                    "dropout_rate": 0.1,
                },
            },
            "best": {
                "description": "Alias for 'mlp' architecture (currently the only fully supported option)",
                "default_params": {
                    "encoder_layers": [128, 256, 256, 128, 64],
                    "decoder_layers": [64, 128, 256, 256, 128],
                    "activation": "relu",
                    "dropout_rate": 0.1,
                },
            },
            "default": {
                "description": "Alias for 'mlp' architecture",
                "default_params": {
                    "encoder_layers": [128, 256, 256, 128, 64],
                    "decoder_layers": [64, 128, 256, 256, 128],
                    "activation": "relu",
                    "dropout_rate": 0.1,
                },
            },
        }

    def train(
        self,
        n_samples=10000,
        batch_size=64,
        n_epochs=50,
        validation_split=0.2,
        encoder_epochs=None,
        decoder_epochs=None,
        end_to_end_epochs=None,
        encoder_lr=None,
        decoder_lr=None,
        end_to_end_lr=None,
        verbose=True,
    ):
        """
        Train the NNAE estimator using a three-step process.

        The training process consists of three phases:
        1. Encoder Training: The encoder learns to predict parameters from function values
        2. Decoder Training: The decoder learns to predict function moments from parameters
        3. End-to-End Training: The complete network is fine-tuned with a tri-component loss

        Args:
            n_samples: Number of synthetic data samples to generate
            batch_size: Batch size for training
            n_epochs: Default number of epochs for all training phases
            validation_split: Fraction of data to use for validation
            encoder_epochs: Number of epochs for encoder training (default: n_epochs // 3)
            decoder_epochs: Number of epochs for decoder training (default: n_epochs // 3)
            end_to_end_epochs: Number of epochs for end-to-end training (default: n_epochs // 3)
            encoder_lr: Learning rate for encoder training (default: self.learning_rate)
            decoder_lr: Learning rate for decoder training (default: self.learning_rate)
            end_to_end_lr: Learning rate for end-to-end training (default: self.learning_rate)
            verbose: Whether to print progress during training

        Returns:
            Dictionary containing training history
        """
        # Set default epochs if not provided
        if encoder_epochs is None:
            encoder_epochs = n_epochs // 3
        if decoder_epochs is None:
            decoder_epochs = n_epochs // 3
        if end_to_end_epochs is None:
            end_to_end_epochs = n_epochs // 3

        # Set default learning rates if not provided
        if encoder_lr is None:
            encoder_lr = self.learning_rate
        if decoder_lr is None:
            decoder_lr = self.learning_rate
        if end_to_end_lr is None:
            end_to_end_lr = self.learning_rate

        print(f"Starting NNAE training with {n_samples} samples")
        print(
            f"Encoder: {encoder_epochs} epochs, "
            f"Decoder: {decoder_epochs} epochs, "
            f"End-to-End: {end_to_end_epochs} epochs"
        )

        # Generate synthetic data using the data generator
        print("Generating synthetic data...")
        data_generator = SyntheticDataGenerator(
            function=self.function,
            param_ranges=self.param_ranges,
            independent_vars_sampling=self.independent_vars_sampling,
        )
        params, function_values = data_generator.generate_dataset(n_samples)

        # Extract function values for the single independent variable case
        if len(self.independent_var_names) == 1:
            var_name = self.independent_var_names[0]
            x_values, y_values = function_values[var_name]
            # y_values is now of shape (n_samples, n_points)
        else:
            raise NotImplementedError("Multiple independent variables not yet supported")

        # Normalize input function values
        print("Normalizing data...")
        self.y_mean = torch.tensor(np.mean(y_values, axis=0), dtype=torch.float32)
        self.y_std = torch.tensor(np.std(y_values, axis=0), dtype=torch.float32)
        y_normalized = (y_values - self.y_mean.numpy()) / self.y_std.numpy()

        # Normalize parameters to [0,1] range
        params_normalized = np.zeros_like(params)
        for i, param_name in enumerate(self.param_ranges.keys()):
            min_val, max_val = self.param_ranges[param_name]
            params_normalized[:, i] = (params[:, i] - min_val) / (max_val - min_val)

        # Split data into training and validation sets
        train_size = int((1 - validation_split) * n_samples)
        y_train, y_val = y_normalized[:train_size], y_normalized[train_size:]
        params_train, params_val = params_normalized[:train_size], params_normalized[train_size:]

        # Create training and validation datasets for encoder
        encoder_train_dataset = TensorDataset(
            torch.tensor(y_train, dtype=torch.float32), torch.tensor(params_train, dtype=torch.float32)
        )
        encoder_val_dataset = TensorDataset(
            torch.tensor(y_val, dtype=torch.float32), torch.tensor(params_val, dtype=torch.float32)
        )

        # Create training and validation datasets for decoder
        decoder_train_dataset = TensorDataset(
            torch.tensor(params_train, dtype=torch.float32),  # Parameters as input to decoder
            torch.tensor(y_train, dtype=torch.float32),  # Function values for moment calculation
        )
        decoder_val_dataset = TensorDataset(
            torch.tensor(params_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
        )

        # Create training and validation datasets for end-to-end training
        end_to_end_train_dataset = TensorDataset(
            torch.tensor(y_train, dtype=torch.float32),  # Function values as input
            torch.tensor(params_train, dtype=torch.float32),  # True parameters for loss calculation
        )
        end_to_end_val_dataset = TensorDataset(
            torch.tensor(y_val, dtype=torch.float32), torch.tensor(params_val, dtype=torch.float32)
        )

        # Create data loaders
        encoder_train_loader = DataLoader(encoder_train_dataset, batch_size=batch_size, shuffle=True)
        encoder_val_loader = DataLoader(encoder_val_dataset, batch_size=batch_size, shuffle=False)

        decoder_train_loader = DataLoader(decoder_train_dataset, batch_size=batch_size, shuffle=True)
        decoder_val_loader = DataLoader(decoder_val_dataset, batch_size=batch_size, shuffle=False)

        end_to_end_train_loader = DataLoader(end_to_end_train_dataset, batch_size=batch_size, shuffle=True)
        end_to_end_val_loader = DataLoader(end_to_end_val_dataset, batch_size=batch_size, shuffle=False)

        # Initialize history dictionary
        history = {
            "encoder_train_loss": [],
            "encoder_val_loss": [],
            "decoder_train_loss": [],
            "decoder_val_loss": [],
            "end_to_end_train_loss": [],
            "end_to_end_val_loss": [],
            "train_reconstruction_loss": [],  # For backward compatibility
            "train_validation_loss": [],  # For backward compatibility
            "val_reconstruction_loss": [],  # For backward compatibility
            "val_validation_loss": [],  # For backward compatibility
        }

        # Step 1: Train encoder
        print("PHASE 1: Training encoder...")
        encoder_history = self._train_encoder(
            self.network.encoder,
            encoder_train_loader,
            encoder_val_loader,
            encoder_epochs,
            encoder_lr,
            self.device,
            verbose,
        )

        # Step 2: Train decoder
        print("PHASE 2: Training decoder...")
        decoder_history = self._train_decoder(
            self.network.decoder,
            decoder_train_loader,
            decoder_val_loader,
            decoder_epochs,
            decoder_lr,
            self.device,
            verbose,
        )

        # Step 3: Train end-to-end
        print("PHASE 3: Training end-to-end...")
        end_to_end_history = self._train_end_to_end(
            self.network,
            end_to_end_train_loader,
            end_to_end_val_loader,
            end_to_end_epochs,
            end_to_end_lr,
            self.device,
            verbose,
        )

        # Update history dictionary
        history["encoder_train_loss"] = encoder_history["train_loss"]
        history["encoder_val_loss"] = encoder_history["val_loss"]
        history["decoder_train_loss"] = decoder_history["train_loss"]
        history["decoder_val_loss"] = decoder_history["val_loss"]
        history["end_to_end_train_loss"] = end_to_end_history["train_loss"]
        history["end_to_end_val_loss"] = end_to_end_history["val_loss"]
        history["train_reconstruction_loss"] = end_to_end_history["train_moment_loss"]
        history["train_validation_loss"] = end_to_end_history["train_param_valid_loss"]
        history["val_reconstruction_loss"] = end_to_end_history["val_moment_loss"]
        history["val_validation_loss"] = end_to_end_history["val_param_valid_loss"]

        # Add standard keys for compatibility with example scripts
        history["train_loss"] = end_to_end_history["train_loss"]
        history["val_loss"] = end_to_end_history["val_loss"]

        print("Training complete!")
        return history

    def _train_encoder(
        self,
        encoder,
        encoder_train_loader,
        encoder_val_loader,
        n_epochs,
        encoder_lr: Optional[float] = None,
        device: torch.device = None,
        verbose: bool = True,
    ):
        """Train the encoder network separately."""
        print("Training encoder...")

        # Move encoder to device
        encoder = encoder.to(device)

        # Set up optimizer
        if encoder_lr is None:
            encoder_lr = self.learning_rate
        optimizer = torch.optim.Adam(encoder.parameters(), lr=encoder_lr)
        # print(f"Encoder optimizer: {optimizer}")
        # print(f"Encoder network: {encoder}")

        # Loss function (mean squared error)
        loss_fn = nn.MSELoss()

        # Training history
        history = {"train_loss": [], "val_loss": []}

        # Training loop
        for epoch in range(n_epochs):
            encoder.train()
            train_loss = 0.0

            # Training loop
            for x_batch, y_batch in encoder_train_loader:
                # Move data to device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)

                # Forward pass
                y_pred = encoder(x_batch)

                # Compute loss
                loss = loss_fn(y_pred, y_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm=1.0)

                # Update weights
                optimizer.step()

                # Update metrics
                train_loss += loss.item() * x_batch.size(0)

            # Calculate average training loss
            train_loss /= len(encoder_train_loader.dataset)
            history["train_loss"].append(train_loss)

            # Log more frequently during initial epochs
            log_interval = 5 if epoch < 20 else 10

            # Validation phase
            if encoder_val_loader:
                encoder.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for x_batch, y_batch in encoder_val_loader:
                        # Move data to device
                        x_batch = x_batch.to(device)
                        y_batch = y_batch.to(device)

                        # Forward pass
                        y_pred = encoder(x_batch)

                        # Compute loss
                        loss = loss_fn(y_pred, y_batch)

                        # Update metrics
                        val_loss += loss.item() * x_batch.size(0)

                # Calculate average validation loss
                val_loss /= len(encoder_val_loader.dataset)
                history["val_loss"].append(val_loss)

                # Log progress
                if verbose and (epoch + 1) % log_interval == 0:
                    print(
                        f"Encoder Epoch {epoch + 1}/{n_epochs} - "
                        f"Loss: {train_loss:.4f} - "
                        f"Val Loss: {val_loss:.4f}"
                    )
            else:
                # Log progress without validation
                if verbose and (epoch + 1) % log_interval == 0:
                    print(f"Encoder Epoch {epoch + 1}/{n_epochs} - " f"Loss: {train_loss:.4f}")

        return history

    def _train_decoder(
        self,
        decoder,
        decoder_train_loader,
        decoder_val_loader,
        n_epochs,
        decoder_lr: Optional[float] = None,
        device: torch.device = None,
        verbose: bool = True,
    ):
        """Train the decoder network separately."""
        print("Training decoder...")

        # Move decoder to device
        decoder = decoder.to(device)

        # Set up optimizer
        if decoder_lr is None:
            decoder_lr = self.learning_rate
        optimizer = torch.optim.Adam(decoder.parameters(), lr=decoder_lr)

        # Loss function (mean squared error)
        loss_fn = nn.MSELoss()

        # Create moment calculator for input data
        moment_calculator = MomentCalculator(n_moments=10).to(device)

        # Training history
        history = {"train_loss": [], "val_loss": []}

        # Training loop
        for epoch in range(n_epochs):
            decoder.train()
            train_loss = 0.0

            # Training loop
            for params_batch, y_batch in decoder_train_loader:
                # Move data to device
                params_batch = params_batch.to(device)
                y_batch = y_batch.to(device)

                # Calculate input moments
                input_moments = moment_calculator.calculate_moments(y_batch)

                # Forward pass
                moments_pred = decoder(params_batch)

                # Compute loss
                loss = loss_fn(moments_pred, input_moments)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=1.0)

                # Update weights
                optimizer.step()

                # Update metrics
                train_loss += loss.item() * params_batch.size(0)

            # Calculate average training loss
            train_loss /= len(decoder_train_loader.dataset)
            history["train_loss"].append(train_loss)

            # Log more frequently during initial epochs
            log_interval = 5 if epoch < 20 else 10

            # Validation phase
            if decoder_val_loader:
                decoder.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for params_batch, y_batch in decoder_val_loader:
                        # Move data to device
                        params_batch = params_batch.to(device)
                        y_batch = y_batch.to(device)

                        # Calculate input moments
                        input_moments = moment_calculator.calculate_moments(y_batch)

                        # Forward pass
                        moments_pred = decoder(params_batch)

                        # Compute loss
                        loss = loss_fn(moments_pred, input_moments)

                        # Update metrics
                        val_loss += loss.item() * params_batch.size(0)

                # Calculate average validation loss
                val_loss /= len(decoder_val_loader.dataset)
                history["val_loss"].append(val_loss)

                # Log progress
                if verbose and (epoch + 1) % log_interval == 0:
                    print(
                        f"Decoder Epoch {epoch + 1}/{n_epochs} - "
                        f"Loss: {train_loss:.4f} - "
                        f"Val Loss: {val_loss:.4f}"
                    )
            else:
                # Log progress without validation
                if verbose and (epoch + 1) % log_interval == 0:
                    print(f"Decoder Epoch {epoch + 1}/{n_epochs} - " f"Loss: {train_loss:.4f}")

        return history

    def _train_end_to_end(
        self,
        network,
        train_loader,
        val_loader,
        n_epochs,
        end_to_end_lr: Optional[float] = None,
        device: torch.device = None,
        verbose: bool = True,
    ):
        """Train the complete network end-to-end."""
        print("Training end-to-end...")

        # Move network to device
        network = network.to(device)

        # Set up optimizer
        if end_to_end_lr is None:
            end_to_end_lr = self.learning_rate
        optimizer = torch.optim.Adam(network.parameters(), lr=end_to_end_lr)

        # Loss function (tri-component NNAE loss)
        loss_fn = _NNAELoss(
            fit_function=self.function,
            indep_vars=self.independent_vars_sampling,
            param_ranges=self.param_ranges,
            alpha=self.alpha,
            beta=self.beta,
        )

        # Create moment calculator for input data
        moment_calculator = MomentCalculator(n_moments=10).to(device)

        # Training history
        history = {
            "train_loss": [],
            "train_moment_loss": [],
            "train_param_valid_loss": [],
            "train_param_accuracy_loss": [],
            "val_loss": [],
            "val_moment_loss": [],
            "val_param_valid_loss": [],
            "val_param_accuracy_loss": [],
        }

        # Training loop
        for epoch in range(n_epochs):
            network.train()
            train_loss = 0.0
            train_moment_loss = 0.0
            train_param_valid_loss = 0.0
            train_param_accuracy_loss = 0.0

            # Training loop
            for x_batch, true_params_batch in train_loader:
                # Move data to device
                x_batch = x_batch.to(device)
                true_params_batch = true_params_batch.to(device)

                # Calculate input moments once for efficiency
                input_moments = moment_calculator.calculate_moments(x_batch)

                # Forward pass - returns params and moments
                params, moments = network(x_batch)

                # Compute loss
                loss, moment_loss, param_valid_loss, param_accuracy_loss = loss_fn(
                    params, moments, true_params_batch, input_moments
                )

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Apply gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)

                # Update weights
                optimizer.step()

                # Update metrics
                batch_size = x_batch.size(0)
                train_loss += loss.item() * batch_size
                train_moment_loss += moment_loss.item() * batch_size
                train_param_valid_loss += param_valid_loss.item() * batch_size
                train_param_accuracy_loss += param_accuracy_loss.item() * batch_size

            # Calculate average training losses
            n_samples = len(train_loader.dataset)
            train_loss /= n_samples
            train_moment_loss /= n_samples
            train_param_valid_loss /= n_samples
            train_param_accuracy_loss /= n_samples

            # Update history
            history["train_loss"].append(train_loss)
            history["train_moment_loss"].append(train_moment_loss)
            history["train_param_valid_loss"].append(train_param_valid_loss)
            history["train_param_accuracy_loss"].append(train_param_accuracy_loss)

            # Log more frequently during initial epochs
            log_interval = 5 if epoch < 20 else 10

            # Validation phase
            if val_loader:
                network.eval()
                val_loss = 0.0
                val_moment_loss = 0.0
                val_param_valid_loss = 0.0
                val_param_accuracy_loss = 0.0

                with torch.no_grad():
                    for x_batch, true_params_batch in val_loader:
                        # Move data to device
                        x_batch = x_batch.to(device)
                        true_params_batch = true_params_batch.to(device)

                        # Calculate input moments once for efficiency
                        input_moments = moment_calculator.calculate_moments(x_batch)

                        # Forward pass
                        params, moments = network(x_batch)

                        # Compute loss
                        loss, moment_loss, param_valid_loss, param_accuracy_loss = loss_fn(
                            params, moments, true_params_batch, input_moments
                        )

                        # Update metrics
                        batch_size = x_batch.size(0)
                        val_loss += loss.item() * batch_size
                        val_moment_loss += moment_loss.item() * batch_size
                        val_param_valid_loss += param_valid_loss.item() * batch_size
                        val_param_accuracy_loss += param_accuracy_loss.item() * batch_size

                # Calculate average validation losses
                n_samples = len(val_loader.dataset)
                val_loss /= n_samples
                val_moment_loss /= n_samples
                val_param_valid_loss /= n_samples
                val_param_accuracy_loss /= n_samples

                # Update history
                history["val_loss"].append(val_loss)
                history["val_moment_loss"].append(val_moment_loss)
                history["val_param_valid_loss"].append(val_param_valid_loss)
                history["val_param_accuracy_loss"].append(val_param_accuracy_loss)

                # Log progress
                if verbose and (epoch + 1) % log_interval == 0:
                    print(
                        f"Epoch {epoch + 1}/{n_epochs} - "
                        f"Loss: {train_loss:.4f} - "
                        f"Moment Loss: {train_moment_loss:.4f} - "
                        f"Param Valid Loss: {train_param_valid_loss:.4f} - "
                        f"Param Accuracy Loss: {train_param_accuracy_loss:.4f} - "
                        f"Val Loss: {val_loss:.4f} - "
                        f"Val Moment Loss: {val_moment_loss:.4f} - "
                        f"Val Param Valid Loss: {val_param_valid_loss:.4f} - "
                        f"Val Param Accuracy Loss: {val_param_accuracy_loss:.4f}"
                    )
            else:
                # Log progress without validation
                if verbose and (epoch + 1) % log_interval == 0:
                    print(
                        f"Epoch {epoch + 1}/{n_epochs} - "
                        f"Loss: {train_loss:.4f} - "
                        f"Moment Loss: {train_moment_loss:.4f} - "
                        f"Param Valid Loss: {train_param_valid_loss:.4f} - "
                        f"Param Accuracy Loss: {train_param_accuracy_loss:.4f}"
                    )

        return history

    def predict(self, x=None, y=None, return_network_output=False):
        """
        Predict parameters for the given function values.

        Args:
            x: Independent variable values (x-data). If provided along with y, y is used as the function values.
            y: Function values (y-data). If provided, this is used for prediction. Required if x is None.
            return_network_output: If True, return both the parameters (normalized parameters) and network outputs.

        Returns:
            If return_network_output is False (default):
                Dict of parameter name to parameter value(s)
            If return_network_output is True:
                Tuple of (parameters dict, network_outputs dict)
        """
        # If y is None but x is provided, use x as the function values (backward compatibility)
        if y is None:
            if x is None:
                raise ValueError("Either x or y must be provided")
            function_values = x
        else:
            function_values = y

        # Convert input to torch tensor if it's not already
        if not isinstance(function_values, torch.Tensor):
            function_values = torch.tensor(function_values, dtype=torch.float32)

        # Add batch dimension if needed
        if function_values.dim() == 1:
            function_values = function_values.unsqueeze(0)

        # Move to device
        function_values = function_values.to(self.device)

        # Normalize input
        function_values_normalized = (function_values - self.y_mean) / self.y_std

        # Forward pass through the network
        self.network.eval()
        with torch.no_grad():
            params, moments = self.network(function_values_normalized)

        # Denormalize parameters
        param_dict = {}
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[param_name]
            param_values = min_val + params[:, i].cpu().numpy() * (max_val - min_val)

            # If batch size is 1, return scalar; otherwise, return array
            if param_values.shape[0] == 1:
                param_dict[param_name] = param_values[0]
            else:
                param_dict[param_name] = param_values

        if return_network_output:
            network_outputs = {"normalized_params": params.cpu().numpy(), "moments": moments.cpu().numpy()}
            return param_dict, network_outputs
        else:
            return param_dict

    def save(self, path: str) -> None:
        """
        Save the trained estimator to a file.

        Args:
            path: Path to save the estimator to
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Prepare state dict
        state_dict = {
            "model_state_dict": self.network.state_dict(),
            "architecture": self.architecture,
            "architecture_params": self.architecture_params,
            "param_ranges": self.param_ranges,
            "independent_vars_sampling": {k: v.tolist() for k, v in self.independent_vars_sampling.items()},
            "alpha": self.alpha,
            "beta": self.beta,
            "y_mean": self.y_mean.cpu().numpy().tolist() if hasattr(self, "y_mean") else None,
            "y_std": self.y_std.cpu().numpy().tolist() if hasattr(self, "y_std") else None,
        }

        # Save to disk
        torch.save(state_dict, path)

    @classmethod
    def load(cls, path: str, function: Callable, device: Optional[str] = None) -> "NNAEEstimator":
        """Load a trained model from disk.

        Args:
            path: Path to load the model from
            function: The curve fitting function (must be provided when loading)
            device: Device to use for computation (default: auto)

        Returns:
            Loaded NNAEEstimator instance
        """
        # Load state dict
        state_dict = torch.load(path, map_location=torch.device("cpu"))

        # Convert independent vars sampling back to numpy arrays
        independent_vars_sampling = {k: np.array(v) for k, v in state_dict["independent_vars_sampling"].items()}

        # Extract architecture parameters
        architecture = state_dict.get("architecture", "mlp")
        architecture_params = state_dict.get("architecture_params", {})
        alpha = state_dict.get("alpha", 0.5)
        beta = state_dict.get("beta", 0.5)

        # Create estimator
        estimator = cls(
            function=function,
            param_ranges=state_dict["param_ranges"],
            independent_vars_sampling=independent_vars_sampling,
            architecture=architecture,
            architecture_params=architecture_params,
            alpha=alpha,
            beta=beta,
            device=device,
        )

        # Load model weights
        estimator.network.load_state_dict(state_dict["model_state_dict"])

        # Load normalization parameters if available
        if state_dict.get("y_mean") is not None and state_dict.get("y_std") is not None:
            estimator.y_mean = torch.tensor(state_dict["y_mean"], dtype=torch.float32)
            estimator.y_std = torch.tensor(state_dict["y_std"], dtype=torch.float32)

        return estimator

    def _extract_y_values(self, *args, **kwargs):
        """
        Extract function values from arguments.

        This method handles various ways that function values could be provided:
        1. Directly as a numpy array or tensor
        2. As keyword arguments matching the fit function's independent variable names
        3. As positional arguments in the order of the fit function's independent variable names

        Returns:
            Function values as a numpy array
        """
        # Case 1: Direct array or tensor as first argument
        if len(args) == 1 and isinstance(args[0], (np.ndarray, torch.Tensor, list)):
            return np.asarray(args[0], dtype=np.float32)

        # Case 2: Function values provided through keyword arguments
        if len(self.independent_var_names) == 1 and self.independent_var_names[0] in kwargs:
            # Single independent variable case
            return np.asarray(kwargs[self.independent_var_names[0]], dtype=np.float32)

        # Case 3: Function values as positional arguments
        if len(args) == len(self.independent_var_names):
            # Multiple independent variables, matched by position
            # For now, we'll just take the first independent variable's values
            # This would need to be extended for more complex scenarios
            return np.asarray(args[0], dtype=np.float32)

        # If we get here, we couldn't find the function values
        raise ValueError(
            f"Could not extract function values from arguments. "
            f"Expected function values as an array, or as keyword arguments "
            f"matching the independent variable names: {self.independent_var_names}"
        )


class MomentCalculator(nn.Module):
    """Utility to calculate the first N central moments of a function using log-scale."""

    def __init__(self, n_moments=10):
        """Initialize the moment calculator.

        Args:
            n_moments: Number of moments to calculate (default: 10)
        """
        super().__init__()
        self.n_moments = n_moments
        self.epsilon = 1e-10  # Small value to prevent division by zero and log(0)

    def calculate_moments(self, y: torch.Tensor) -> torch.Tensor:
        """Calculate the first n_moments central moments of the function in log-scale.

        Args:
            y: Function values [batch_size, function_length]

        Returns:
            Tensor of shape [batch_size, n_moments] containing the log-scale moments
        """
        batch_size = y.shape[0]
        moments = torch.zeros((batch_size, self.n_moments), device=y.device)

        # For each sample in the batch
        for i in range(batch_size):
            y_sample = y[i]

            # Mean (first moment)
            mean = torch.mean(y_sample)
            std = torch.std(y_sample) + self.epsilon

            # Store log-scaled mean with sign preservation
            # log(|x| + ε) * sign(x)
            mean_sign = torch.sign(mean)
            log_abs_mean = torch.log(torch.abs(mean) + self.epsilon)
            moments[i, 0] = log_abs_mean * mean_sign

            # Central moments (2 to n_moments)
            for j in range(1, self.n_moments):
                # Calculate raw central moment
                central_moment = torch.mean((y_sample - mean) ** (j + 1))

                # Normalize by appropriate power of standard deviation
                normalized_moment = central_moment / (std ** (j + 1) + self.epsilon)

                # Apply log-scale with sign preservation
                moment_sign = torch.sign(normalized_moment)
                log_abs_moment = torch.log(torch.abs(normalized_moment) + self.epsilon)

                # Store the log-scale moment
                moments[i, j] = log_abs_moment * moment_sign

                # Apply additional normalization to keep values in a reasonable range
                # Clip to prevent extreme values
                moments[i, j] = torch.clamp(moments[i, j], -10.0, 10.0)

        return moments

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        """Forward pass to calculate log-scale moments.

        Args:
            y: Function values [batch_size, function_length]

        Returns:
            Tensor of shape [batch_size, n_moments] containing the log-scale moments
        """
        return self.calculate_moments(y)


class _NNAENetwork(nn.Module):
    """
    Neural Network Autoencoder for parameter estimation.

    The network consists of two main components:
    1. Encoder: Maps function values to parameter estimates
    2. Decoder: Maps parameter estimates to function statistical moments

    This architecture is designed to learn an efficient representation of the function's
    parameters by encoding the essential statistical information from the function values.
    """

    def __init__(self, input_size, n_params, encoder_layers=None, decoder_layers=None, n_moments=10):
        """
        Initialize the NNAE network.

        Args:
            input_size: Number of input points in the function (function dimension)
            n_params: Number of parameters to estimate
            encoder_layers: List of hidden layer sizes for the encoder (default: [128, 256, 256, 128, 64])
            decoder_layers: List of hidden layer sizes for the decoder (default: [64, 128, 256, 256, 128])
            n_moments: Number of statistical moments to output from decoder
        """
        super().__init__()
        self.input_size = input_size
        self.n_params = n_params
        self.n_moments = n_moments

        # Set default encoder/decoder architectures if not provided
        if encoder_layers is None:
            encoder_layers = [128, 256, 256, 128, 64]
        if decoder_layers is None:
            decoder_layers = [64, 128, 256, 256, 128]

        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers

        # Create encoder network that maps function values to parameters
        self.encoder = self._create_encoder(input_size, n_params, encoder_layers)

        # Create decoder network that maps parameters to function moments
        self.decoder = _ResidualDecoder(n_params, n_moments, decoder_layers)

    def _create_encoder(self, input_size, n_params, hidden_layers):
        """Create the encoder network architecture."""
        layers = []

        # Input layer
        prev_size = input_size

        # Hidden layers
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(size))
            prev_size = size

        # Output layer with sigmoid to normalize parameters to [0,1]
        layers.append(nn.Linear(prev_size, n_params))
        layers.append(nn.Sigmoid())

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the NNAE network.

        Args:
            x: Input function values

        Returns:
            Tuple of (normalized_params, moments)
        """
        # Pass through encoder to get normalized parameters
        normalized_params = self.encoder(x)

        # Pass through decoder to get function moments
        moments = self.decoder(normalized_params)

        return normalized_params, moments


class _ResidualDecoder(nn.Module):
    """
    Enhanced decoder network that maps parameters to function moments.

    This decoder uses:
    - Residual connections to improve gradient flow
    - Parallel processing paths for different parameter transformations
    - Batch normalization for training stability
    - Outputs the first 10 central moments of the function rather than
      reconstructing the full function values directly.
    """

    def __init__(self, param_dim, n_moments=10, decoder_layers=None):
        super().__init__()

        self.param_dim = param_dim
        self.n_moments = n_moments  # Number of moments to predict

        # Set default decoder architecture if not provided
        if decoder_layers is None:
            decoder_layers = [64, 128, 256, 256, 128]

        self.decoder_layers = decoder_layers

        # Create the main decoder network
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Input projection
        input_size = decoder_layers[0]
        self.input_projection = nn.Linear(param_dim, input_size)
        self.input_norm = nn.BatchNorm1d(input_size)

        # Create the hidden layers with batch normalization
        for i in range(len(decoder_layers) - 1):
            self.layers.append(nn.Linear(decoder_layers[i], decoder_layers[i + 1]))
            self.batch_norms.append(nn.BatchNorm1d(decoder_layers[i + 1]))
            self.dropouts.append(nn.Dropout(0.1))

        # Final output layer produces n_moments moments
        self.output_layer = nn.Linear(decoder_layers[-1], n_moments)

        # Additional modules for residual connections
        self.residual_projections = nn.ModuleList()
        for i in range(len(decoder_layers) - 1):
            self.residual_projections.append(nn.Linear(param_dim, decoder_layers[i + 1]))

        # Additional feature pathway for complex transformations
        self.complex_path = nn.Sequential(
            nn.Linear(param_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_moments),  # Output n_moments
        )

    def forward(self, x):
        # Main branch - initial projection
        main = self.input_projection(x)
        main = self.input_norm(main)
        main = nn.functional.relu(main)

        # Process through hidden layers with residual connections
        for _, (layer, bn, dropout, res_proj) in enumerate(
            zip(self.layers, self.batch_norms, self.dropouts, self.residual_projections)
        ):
            # Skip connection from input parameters
            residual = res_proj(x)

            # Main path
            main = layer(main)
            main = bn(main)
            main = nn.functional.relu(main)
            main = dropout(main)

            # Add residual
            main = main + residual

        # Final layer to produce moments
        main_output = self.output_layer(main)

        # Parallel path for complex transformations
        complex_output = self.complex_path(x)

        # Combine outputs from both paths for final moments prediction
        moments = main_output + complex_output

        # Note: We do not apply any specific activation to moments as they can be any real number
        return moments


class _NNAELoss:
    """
    Tri-component loss function for NNAE training.

    The loss consists of three components:
    1. L₁: Moment loss - MSE between predicted and target moments
    2. L₂: Parameter validation loss - validates parameter outputs using the fit function
    3. L₃: Parameter accuracy loss - MSE between predicted and true parameters

    The combined loss is calculated as α·L₁·L₃ + β·L₂
    """

    def __init__(self, fit_function, indep_vars, param_ranges, alpha=1.0, beta=0.5):
        """
        Initialize the NNAE loss function.

        Args:
            fit_function: The function to fit parameters to
            indep_vars: Independent variables for the fit function
            param_ranges: Dictionary of parameter ranges {param_name: (min, max)}
            alpha: Weight for the moment loss and parameter accuracy components
            beta: Weight for the parameter validation loss component
        """
        self.fit_function = fit_function
        self.indep_vars = indep_vars
        self.param_ranges = param_ranges
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()

    def __call__(self, params, moments, true_params, input_moments):
        """
        Compute the NNAE loss.

        Args:
            params: Predicted parameters from the encoder
            moments: Predicted moments from the decoder
            true_params: Ground truth parameters (for training)
            input_moments: Input moments calculated from input y values

        Returns:
            Tuple of (total_loss, moment_loss, param_valid_loss, param_accuracy_loss)
        """
        # L₁: Moment Reconstruction Loss
        moment_loss = self.mse(moments, input_moments)

        # L₂: Parameter Validation Loss
        param_valid_loss = self._compute_parameter_validation_loss(params, moments)

        # L₃: Parameter Accuracy Loss
        param_accuracy_loss = self.mse(params, true_params)

        # Combined loss: α·L₁·L₃ + β·L₂
        total_loss = self.alpha * moment_loss * param_accuracy_loss + self.beta * param_valid_loss

        return total_loss, moment_loss, param_valid_loss, param_accuracy_loss

    def _compute_parameter_validation_loss(self, params, moments):
        """
        Compute parameter validation loss (L₂) by comparing decoder moments with
        moments calculated from function values generated using the predicted parameters.

        This validates that the decoder's predicted moments match the actual moments
        of the function when using the estimated parameters.

        Args:
            params: Predicted parameters from the encoder (normalized to [0,1])
            moments: Moments predicted by the decoder

        Returns:
            loss: Parameter validation loss (MSE between predicted moments and moments
                  calculated from function values using predicted parameters)
        """
        batch_size = params.shape[0]
        device = params.device

        # Initialize an array to store the function moments for each sample in the batch
        computed_moments = torch.zeros_like(moments)

        # Unnormalize parameters
        unnormalized_params = {}
        for i, param_name in enumerate(self.param_ranges.keys()):
            min_val, max_val = self.param_ranges[param_name]
            param_values = params[:, i].detach().cpu().numpy() * (max_val - min_val) + min_val
            unnormalized_params[param_name] = param_values

        # Create moment calculator
        moment_calculator = MomentCalculator(n_moments=moments.shape[1]).to(device)

        # Loop through each sample in the batch
        for i in range(batch_size):
            try:
                # Extract parameters for this sample
                sample_params = {k: v[i] for k, v in unnormalized_params.items()}

                # Calculate function values using these parameters
                x_values = next(iter(self.indep_vars.values()))  # Get the first independent variable's values
                y_values = self.fit_function(x_values, **sample_params)

                # Convert to tensor and move to the correct device
                y_tensor = torch.tensor(y_values, dtype=torch.float32, device=device).unsqueeze(0)

                # Calculate moments from these function values
                computed_moments[i] = moment_calculator(y_tensor)

            except Exception:
                # If there's an error, set computed moments to zeros
                # This will naturally create a high loss for invalid parameters
                pass

        # Calculate MSE between predicted moments and computed moments
        loss = self.mse(moments, computed_moments)

        return loss
