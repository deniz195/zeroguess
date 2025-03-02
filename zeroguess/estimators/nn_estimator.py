"""
Neural network-based parameter estimator implementation.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from zeroguess.data.generators import SyntheticDataGenerator
from zeroguess.estimators.architectures import get_architecture, get_architecture_info, list_architectures
from zeroguess.estimators.base import BaseEstimator


class NeuralNetworkEstimator(BaseEstimator):
    """Neural network-based parameter estimator.

    This estimator uses a neural network to map from function inputs to parameter estimates.
    It supports different neural network architectures that can be selected during initialization.

    Available architectures:
    - "mlp" (default): Multilayer Perceptron with fully connected layers
    - "cnn": Convolutional Neural Network (future work)
    - "transformer": Transformer network with self-attention (future work)

    You can also use "best" or "default" to use the recommended architecture.

    Hardware acceleration is automatically selected in the following order:
    1. CUDA (NVIDIA GPUs)
    2. MPS (Apple Silicon)
    3. CPU (fallback)

    You can manually override this selection with the device parameter.
    """

    def __init__(
        self,
        function: Callable,
        param_ranges: Dict[str, Tuple[float, float]],
        independent_vars_sampling: Dict[str, np.ndarray],
        architecture: str = "best",
        architecture_params: Optional[Dict[str, Any]] = None,
        learning_rate: float = 0.001,
        weight_decay: float = 0.00001,
        device: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the neural network estimator.

        Args:
            function: The curve fitting target function
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            independent_vars_sampling: Dictionary mapping independent variable names
                to arrays of sampling points
            architecture: Architecture type to use (default: "best")
                Options include: "mlp" (default), "cnn" (future), "transformer" (future)
                You can also use "best" or "default" to use the recommended architecture.
            architecture_params: Architecture-specific parameters
                Use get_architecture_info() to see available parameters for each architecture
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            device: Device to use for computation (default: auto)
                Options: "cuda", "mps", "cpu", or None (auto-detect)
            **kwargs: Additional keyword arguments
        """
        super().__init__(function, param_ranges, independent_vars_sampling, **kwargs)

        # Handle architecture parameters
        if architecture_params is None:
            architecture_params = {}

        self.architecture_name = architecture
        self.architecture_params = architecture_params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # Set up device selection
        if device is None:
            # Auto-detect best available device in priority order: CUDA > MPS > CPU
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            # Use specified device
            self.device = torch.device(device)

            # Validate device selection when manually specified
            if device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                print("Warning: MPS requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")

        # These will be initialized during training
        self.network = None
        self.architecture = None
        self.data_generator = None
        self.scaler_x = None
        self.scaler_y = None

        # Instantiate the architecture
        self._create_architecture()

    def _create_architecture(self):
        """Create the specified neural network architecture."""
        self.architecture = get_architecture(self.architecture_name, **self.architecture_params)

    @staticmethod
    def list_available_architectures() -> List[str]:
        """List all available neural network architectures.

        Returns:
            List of available architecture names
        """
        return list_architectures()

    @staticmethod
    def get_architecture_details() -> Dict[str, Dict[str, Any]]:
        """Get details about all available architectures.

        Returns:
            Dictionary mapping architecture names to information dictionaries
            containing description and default parameters
        """
        return get_architecture_info()

    def train(  # noqa: C901
        self,
        n_samples: int = 1000,
        batch_size: int = 64,
        n_epochs: int = 100,
        validation_split: float = 0.2,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the neural network on synthetic data.

        Args:
            n_samples: Number of synthetic samples to generate
            batch_size: Batch size for training
            n_epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            verbose: Whether to print progress during training
            **kwargs: Additional keyword arguments

        Returns:
            Dictionary containing training history and metrics

        Note:
            If training is interrupted by keyboard interrupt (Ctrl+C),
            the model will save its current state and return the
            training history up to that point.
        """

        # Create data generator if it doesn't exist
        if self.data_generator is None:
            self.data_generator = SyntheticDataGenerator(
                function=self.function,
                param_ranges=self.param_ranges,
                independent_vars_sampling=self.independent_vars_sampling,
            )

        # Initialize the history dictionary with empty lists for loss metrics
        history = {"train_loss": [], "val_loss": []}

        try:
            # Generate synthetic training data
            params, function_values = self.data_generator.generate_dataset(n_samples=n_samples)

            # Process the data for training
            # For now, we'll assume a single independent variable scenario
            # In the future, this could be extended for multiple independent variables
            if len(self.independent_var_names) == 1:
                var_name = self.independent_var_names[0]
                _, y_values = function_values[var_name]

                # Flatten the data for training
                X = y_values.reshape(n_samples, -1)  # Shape: (n_samples, n_points)

                # Normalize parameters to [0, 1] range for training
                y_normalized = np.zeros_like(params)
                for i, name in enumerate(self.param_names):
                    min_val, max_val = self.param_ranges[name]
                    y_normalized[:, i] = (params[:, i] - min_val) / (max_val - min_val)

                y = y_normalized  # Use normalized parameters for training
            else:
                raise NotImplementedError("Multiple independent variables not yet implemented")

            # Split data into training and validation sets
            n_val = int(n_samples * validation_split)
            n_train = n_samples - n_val

            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X = X[indices]
            y = y[indices]

            X_train, X_val = X[:n_train], X[n_train:]
            y_train, y_val = y[:n_train], y[n_train:]

            # Create PyTorch datasets and dataloaders
            train_dataset = TensorDataset(
                torch.tensor(X_train, dtype=torch.float32),
                torch.tensor(y_train, dtype=torch.float32),
            )
            val_dataset = TensorDataset(
                torch.tensor(X_val, dtype=torch.float32),
                torch.tensor(y_val, dtype=torch.float32),
            )

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

            # Create the neural network if it doesn't exist
            if self.network is None:
                n_input_features = X.shape[1]
                n_output_params = len(self.param_names)

                # Create the network using the selected architecture
                self.network = self.architecture.create_network(
                    n_input_features=n_input_features, n_output_params=n_output_params
                )

                # Move the network to the appropriate device
                self.network.to(self.device)

            # Create optimizer
            optimizer = optim.Adam(
                self.network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            # print(f"Encoder optimizer: {optimizer}")
            # print(f"Network: {self.network}")

            # Define loss function
            criterion = nn.MSELoss()

            print(
                f"Starting training on {self.architecture_name} architecture "
                f"with {n_epochs} epochs, {n_samples} samples, {batch_size} batch size "
                f"on {self.device} device"
            )

            # Training loop
            completed_epochs = 0

            for epoch in range(n_epochs):
                # Training phase
                self.network.train()
                train_loss = 0.0

                for X_batch, y_batch in train_loader:
                    X_batch = X_batch.to(self.device)
                    y_batch = y_batch.to(self.device)

                    # Forward pass
                    y_pred = self.network(X_batch)
                    loss = criterion(y_pred, y_batch)

                    # Backward pass and optimization
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item() * X_batch.size(0)

                train_loss /= len(train_loader.dataset)

                # Validation phase
                self.network.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)

                        y_pred = self.network(X_batch)
                        loss = criterion(y_pred, y_batch)

                        val_loss += loss.item() * X_batch.size(0)

                    val_loss /= len(val_loader.dataset)

                # Record history
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)

                # Update completed epochs count
                completed_epochs = epoch + 1

                # Print progress
                if verbose and (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1}/{n_epochs} - "
                        f"train_loss: {train_loss:.6f} - "
                        f"val_loss: {val_loss:.6f}"
                    )

        except KeyboardInterrupt:
            if verbose:
                print(f"\nTraining interrupted at epoch {completed_epochs}/{n_epochs}")

                # Only print metrics if we've completed at least one epoch
                if completed_epochs > 0:
                    train_loss = history["train_loss"][-1]
                    val_loss = history["val_loss"][-1]
                    print(f"Latest metrics - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}")

                print("Model saved with current state.")

            # Additional information in history to indicate training was interrupted
            history["interrupted"] = True
            history["completed_epochs"] = completed_epochs

        # Mark as trained regardless of whether we completed all epochs or were interrupted
        # (as long as the network was created)
        self.is_trained = self.network is not None

        # If no epochs were completed, indicate this in the history
        if "completed_epochs" not in history:
            history["completed_epochs"] = n_epochs

        return history

    def predict(self, *args, **kwargs) -> Dict[str, float]:
        """Predict initial parameters for a function.

        Args:
            *args: Positional arguments to pass to the function
            **kwargs: Keyword arguments to pass to the function

        Returns:
            Dictionary mapping parameter names to predicted values

        Raises:
            RuntimeError: If the estimator is not trained
            ValueError: If the input size doesn't match what the network expects
        """
        if not self.is_trained or self.network is None:
            raise RuntimeError("Estimator must be trained before prediction")

        # Handle input data differently based on the form (args vs kwargs)
        if len(args) > 0:
            # Extract independent variable data from args
            if len(args) != 1 and len(self.independent_var_names) == 1:
                # If we expected y values directly (special case for examples)
                if len(args) == 2 or "y" in kwargs:
                    # Extract y values (function output) from the second argument or kwargs
                    args[0]
                    y_data = args[1] if len(args) > 1 else kwargs.get("y")
                else:
                    raise ValueError(
                        f"Expected {len(self.independent_var_names)} positional arguments, " f"got {len(args)}"
                    )
            else:
                # Normal case: extract data from args in expected order
                args[0]
                y_data = kwargs.get("y", None)

                if y_data is None:
                    raise ValueError("y data must be provided for prediction")
        else:
            # Extract independent variable data from kwargs
            if len(self.independent_var_names) == 1:
                var_name = self.independent_var_names[0]
                if var_name not in kwargs:
                    raise ValueError(f"Missing required independent variable: {var_name}")
                kwargs[var_name]
                y_data = kwargs.get("y", None)

                if y_data is None:
                    raise ValueError("y data must be provided for prediction")
            else:
                raise NotImplementedError("Multiple independent variables not yet implemented")

        # Check the network input size from the state dictionary
        # Get the first layer's weight shape
        first_layer_params = next(iter(self.network.parameters()))
        expected_input_size = first_layer_params.shape[1]

        # Use the architecture's input validation method
        self.architecture.validate_input_size(
            network=self.network,
            input_size=len(y_data),
            expected_size=expected_input_size,
        )

        # Convert y_data to tensor and prepare for model input
        features = y_data.flatten()

        # Convert to tensor and move to device
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Make prediction
        self.network.eval()
        with torch.no_grad():
            # Get normalized predictions from network
            predictions_normalized = self.network(features_tensor).cpu().numpy()[0]

            # Ensure predictions are clipped to [0, 1] range before denormalizing
            predictions_normalized = np.clip(predictions_normalized, 0.0, 1.0)

        # Convert predictions back to parameter dictionary with denormalization
        param_dict = {}
        for i, name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[name]
            # Denormalize predictions to the original parameter range
            param_dict[name] = min_val + predictions_normalized[i] * (max_val - min_val)

        return param_dict

    def save(self, path: str) -> None:
        """Save the trained model to disk.

        Args:
            path: Path to save the model

        Raises:
            RuntimeError: If the estimator is not trained
        """
        if not self.is_trained or self.network is None:
            raise RuntimeError("Estimator must be trained before saving")

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

        # Save the model state
        state = {
            "architecture_name": self.architecture_name,
            "architecture_params": self.architecture_params,
            "param_ranges": self.param_ranges,
            "param_names": self.param_names,
            "independent_var_names": self.independent_var_names,
            "independent_vars_sampling": self.independent_vars_sampling,
            "network_state_dict": self.network.state_dict(),
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

        torch.save(state, path)

    @classmethod
    def load(cls, path: str, device: Optional[str] = None) -> "NeuralNetworkEstimator":
        """Load a trained model from disk.

        Args:
            path: Path to load the model from
            device: Device to use for computation (default: auto)
                Options: "cuda", "mps", "cpu", or None (auto-detect)

        Returns:
            Loaded NeuralNetworkEstimator instance

        Raises:
            FileNotFoundError: If the model file does not exist
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model file not found: {path}")

        # Determine device to load model onto
        if device is None:
            # Auto-detect best available device
            if torch.cuda.is_available():
                device_obj = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_obj = torch.device("mps")
            else:
                device_obj = torch.device("cpu")
        else:
            # Use specified device with validation
            if device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                device_obj = torch.device("cpu")
            elif device == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
                print("Warning: MPS requested but not available. Falling back to CPU.")
                device_obj = torch.device("cpu")
            else:
                device_obj = torch.device(device)

        # Load the model state
        state = torch.load(path, map_location=device_obj)

        # Extract the independent_vars_sampling from the state, or create a dummy one
        # that will pass validation in the base class
        independent_vars_sampling = state.get("independent_vars_sampling", None)
        if independent_vars_sampling is None:
            # Create dummy sampling for validation
            independent_vars_sampling = {}
            for var_name in state["independent_var_names"]:
                independent_vars_sampling[var_name] = np.array([0.0])  # Dummy value that will pass validation

        # Create a new estimator instance
        estimator = cls(
            function=None,  # Function is not necessary for prediction
            param_ranges=state["param_ranges"],
            independent_vars_sampling=independent_vars_sampling,
            architecture=state["architecture_name"],
            architecture_params=state["architecture_params"],
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
            device=device,  # Pass through the device parameter
        )

        # Set additional attributes
        estimator.param_names = state["param_names"]
        estimator.independent_var_names = state["independent_var_names"]

        # Check the network input size from the state dictionary
        # Get the first layer's weight shape from the network state dict
        first_layer_key = [k for k in state["network_state_dict"].keys() if "weight" in k][0]
        n_input_features = state["network_state_dict"][first_layer_key].shape[1]

        n_output_params = len(estimator.param_names)

        # Create the network using the architecture
        estimator.network = estimator.architecture.create_network(
            n_input_features=n_input_features, n_output_params=n_output_params
        )

        # Load the network state
        estimator.network.load_state_dict(state["network_state_dict"])
        estimator.network.to(estimator.device)
        estimator.network.eval()

        estimator.is_trained = True

        return estimator
