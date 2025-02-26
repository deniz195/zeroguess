"""
Neural network-based parameter estimator implementation.
"""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from zeroguess.estimators.base import BaseEstimator
from zeroguess.data.generators import SyntheticDataGenerator


class ParameterEstimationNetwork(nn.Module):
    """Neural network for parameter estimation."""
    
    def __init__(
        self,
        n_input_features: int,
        n_output_params: int,
        hidden_layers: List[int] = [128, 256, 256, 256, 128, 64, 32],
    ):
        """Initialize the neural network.
        
        Args:
            n_input_features: Number of input features (typically the number of data points)
            n_output_params: Number of output parameters to estimate
            hidden_layers: List of hidden layer sizes
        """
        super().__init__()

        print(f"Initializing network with {n_input_features} input features and {n_output_params} output parameters")
        
        # Create the network layers
        layers = []
        prev_size = n_input_features
        
        # Add hidden layers
        for size in hidden_layers:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size
        
        # Add output layer
        layers.append(nn.Linear(prev_size, n_output_params))
        
        # Create the sequential model
        self.model = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, n_input_features)
            
        Returns:
            Output tensor of shape (batch_size, n_output_params)
        """
        return self.model(x)


class NeuralNetworkEstimator(BaseEstimator):
    """Neural network-based parameter estimator."""
    
    def __init__(
        self,
        function: Callable,
        param_ranges: Dict[str, Tuple[float, float]],
        independent_vars_sampling: Dict[str, np.ndarray],
        hidden_layers: List[int] = [128, 64, 32],
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0001,
        **kwargs
    ):
        """Initialize the neural network estimator.
        
        Args:
            function: The curve fitting target function
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            independent_vars_sampling: Dictionary mapping independent variable names
                to arrays of sampling points
            hidden_layers: List of hidden layer sizes for the neural network
            learning_rate: Learning rate for the optimizer
            weight_decay: Weight decay for the optimizer
            **kwargs: Additional keyword arguments
        """
        super().__init__(function, param_ranges, independent_vars_sampling, **kwargs)
        
        self.hidden_layers = hidden_layers
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # These will be initialized during training
        self.network = None
        self.data_generator = None
        self.scaler_x = None
        self.scaler_y = None
    
    def _initialize_network(self, input_size: int) -> None:
        """Initialize the neural network.
        
        Args:
            input_size: Size of the input features
        """
        self.network = ParameterEstimationNetwork(
            n_input_features=input_size,
            n_output_params=len(self.param_names),
            hidden_layers=self.hidden_layers,
        ).to(self.device)
    
    def _normalize_parameters(self, params: np.ndarray) -> np.ndarray:
        """Normalize parameters to [0, 1] range based on param_ranges.
        
        Args:
            params: Array of parameter values
            
        Returns:
            Normalized parameters
        """
        normalized = np.zeros_like(params)
        
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[param_name]
            normalized[:, i] = (params[:, i] - min_val) / (max_val - min_val)
        
        return normalized
    
    def _denormalize_parameters(self, normalized_params: np.ndarray) -> np.ndarray:
        """Denormalize parameters from [0, 1] range to original range.
        
        Args:
            normalized_params: Normalized parameter values
            
        Returns:
            Denormalized parameters clipped to their bounds
        """
        denormalized = np.zeros_like(normalized_params)
        
        for i, param_name in enumerate(self.param_names):
            min_val, max_val = self.param_ranges[param_name]
            # First, clip the normalized parameters to [0, 1] range to ensure
            # they are in the valid normalized range
            clipped_norm = np.clip(normalized_params[:, i], 0.0, 1.0)
            # Then denormalize to the original range
            denormalized[:, i] = clipped_norm * (max_val - min_val) + min_val
        
        return denormalized
    
    def train(
        self,
        n_samples: int = 1000,
        batch_size: int = 32,
        epochs: int = 100,
        validation_split: float = 0.2,
        add_noise: bool = True,
        noise_level: float = 0.05,
        **kwargs
    ) -> Dict[str, Any]:
        """Train the neural network on synthetic data.
        
        Args:
            n_samples: Number of synthetic data samples to generate
            batch_size: Batch size for training
            epochs: Number of training epochs
            validation_split: Fraction of data to use for validation
            add_noise: Whether to add noise to the training data
            noise_level: Level of noise to add (if add_noise is True)
            **kwargs: Additional training parameters
            
        Returns:
            Dictionary containing training metrics
        """
        # Initialize data generator
        self.data_generator = SyntheticDataGenerator(
            function=self.function,
            param_ranges=self.param_ranges,
            independent_vars_sampling=self.independent_vars_sampling,
        )
        
        # Generate synthetic dataset
        params, function_values = self.data_generator.generate_dataset(
            n_samples=n_samples,
            add_noise=add_noise,
            noise_level=noise_level,
        )
        
        # For now, assume single independent variable for simplicity
        if len(self.independent_var_names) != 1:
            raise NotImplementedError("Only single independent variable is supported for now")
        
        var_name = self.independent_var_names[0]
        _, y_values = function_values[var_name]
        
        # Prepare data for training
        X = y_values  # The network input is the function values (curve)
        y = self._normalize_parameters(params)  # Target is the normalized parameters
        
        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        
        # Create dataset and dataloaders
        dataset = TensorDataset(X_tensor, y_tensor)
        
        # Split into training and validation
        val_size = int(validation_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # Initialize network (input size is the number of data points in the curve)
        self._initialize_network(X.shape[1])
        
        # Initialize optimizer and loss function
        optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        criterion = nn.MSELoss()
        
        # Training loop
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.network.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.network(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
            
            train_loss /= len(train_loader.dataset)
            train_losses.append(train_loss)
            
            # Validation phase
            self.network.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    targets = targets.to(self.device)
                    
                    outputs = self.network(inputs)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * inputs.size(0)
            
            val_loss /= len(val_loader.dataset)
            val_losses.append(val_loss)
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        self.is_trained = True
        
        return {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "final_train_loss": train_losses[-1],
            "final_val_loss": val_losses[-1],
            "epochs": epochs,
            "n_samples": n_samples,
        }
    
    def predict(self, x_data: np.ndarray, y_data: np.ndarray) -> Dict[str, float]:
        """Predict parameter values for the given data.
        
        Args:
            x_data: Independent variable values
            y_data: Dependent variable values (the curve to fit)
            
        Returns:
            Dictionary mapping parameter names to estimated values,
            clipped to stay within the specified parameter ranges
        """
        if not self.is_trained:
            raise RuntimeError("Estimator must be trained before prediction")
        
        # Convert y_data to the right shape for the network
        # Resample if necessary to match the training input size
        if len(self.independent_var_names) != 1:
            raise NotImplementedError("Only single independent variable is supported for now")
        
        var_name = self.independent_var_names[0]
        training_x = self.independent_vars_sampling[var_name]
        
        # Check if x_data matches the training sampling points
        if x_data.shape != training_x.shape or not np.allclose(x_data, training_x):
            # Resample y_data to match training_x
            # This is a simplistic approach - could be improved with interpolation
            raise NotImplementedError(
                "Input x_data must match the training sampling points for now"
            )
        
        # Prepare input for the network
        X = y_data.reshape(1, -1)  # Add batch dimension
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        # Get prediction from the network
        self.network.eval()
        with torch.no_grad():
            normalized_params = self.network(X_tensor).cpu().numpy()
        
        # Denormalize parameters
        params = self._denormalize_parameters(normalized_params)[0]  # Remove batch dimension
        
        # Create dictionary of parameter name to value
        result = {}
        for i, name in enumerate(self.param_names):
            # Get parameter value
            param_value = params[i]
            # Apply final clipping to ensure bounds are respected
            min_val, max_val = self.param_ranges[name]
            clipped_value = np.clip(param_value, min_val, max_val)
            result[name] = clipped_value
        
        return result
    
    def save(self, path: str) -> None:
        """Save the trained estimator to a file.
        
        Args:
            path: Path to save the estimator to
        """
        if not self.is_trained:
            raise RuntimeError("Cannot save an untrained estimator")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Save model state and metadata
        state = {
            "network_state": self.network.state_dict(),
            "param_names": self.param_names,
            "param_ranges": self.param_ranges,
            "independent_var_names": self.independent_var_names,
            "independent_vars_sampling": {
                k: v.tolist() for k, v in self.independent_vars_sampling.items()
            },
            "hidden_layers": self.hidden_layers,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }
        
        torch.save(state, path)
    
    @classmethod
    def load(cls, path: str) -> 'NeuralNetworkEstimator':
        """Load a trained estimator from a file.
        
        Args:
            path: Path to load the estimator from
            
        Returns:
            Loaded estimator instance
        """
        # Load state dictionary
        state = torch.load(path, map_location=torch.device("cpu"))
        
        # Convert sampling points back to numpy arrays
        independent_vars_sampling = {
            k: np.array(v) for k, v in state["independent_vars_sampling"].items()
        }
        
        # Create a dummy function for initialization
        # This will be replaced with the actual function later
        def dummy_function(*args, **kwargs):
            return np.zeros(10)
        
        # Create instance
        estimator = cls(
            function=dummy_function,
            param_ranges=state["param_ranges"],
            independent_vars_sampling=independent_vars_sampling,
            hidden_layers=state["hidden_layers"],
            learning_rate=state["learning_rate"],
            weight_decay=state["weight_decay"],
        )
        
        # Initialize network
        input_size = list(independent_vars_sampling.values())[0].shape[0]
        estimator._initialize_network(input_size)
        
        # Load network state
        estimator.network.load_state_dict(state["network_state"])
        estimator.is_trained = True
        
        return estimator
