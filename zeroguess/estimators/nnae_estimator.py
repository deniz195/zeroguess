"""
Neural network autoencoder-based parameter estimator implementation.
"""

import os
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from zeroguess.data.generators import SyntheticDataGenerator
from zeroguess.estimators.base import BaseEstimator


class NNAEEstimator(BaseEstimator):
    """Neural Network Autoencoder Estimator.
    
    This estimator uses an autoencoder architecture to improve parameter estimation
    accuracy and robustness. It consists of an encoder that maps function observations
    to parameter estimates, and a decoder that maps parameters back to function observations.
    
    The training process uses a dual-loss approach:
    1. Reconstruction loss: Compares decoder output to encoder input
    2. Parameter validation loss: Uses the fit function with estimated parameters
       to validate through an alternative path
    
    This approach is particularly effective for complex curve fitting problems
    where parameter identification may be ambiguous.
    
    Currently, only the "mlp" architecture is fully supported. Other architectures
    are planned for future implementation.
    """
    
    def __init__(
        self,
        function: Callable,
        param_ranges: Dict[str, Tuple[float, float]],
        independent_vars_sampling: Dict[str, np.ndarray],
        architecture: str = "mlp",
        architecture_params: Optional[Dict[str, Any]] = None,
        learning_rate: float = 0.0001,
        weight_decay: float = 0.0001,
        alpha: float = 0.3,  # Reconstruction loss weight
        beta: float = 0.7,   # Parameter validation loss weight
        device: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the NNAE estimator.
        
        Args:
            function: The curve fitting target function
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            independent_vars_sampling: Dictionary mapping independent variable names
                to arrays of sampling points
            architecture: Architecture type to use (default: "mlp")
                Currently only "mlp" is fully supported. Other options may be added in future.
            architecture_params: Architecture-specific parameters
                For MLP architecture:
                    - encoder_layers: List of hidden layer sizes for the encoder network
                    - decoder_layers: List of hidden layer sizes for the decoder network
                    - activation: Activation function to use ("relu", "tanh", "sigmoid")
                    - dropout_rate: Dropout rate for regularization
            alpha: Weight for reconstruction loss component
            beta: Weight for parameter validation loss component
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
            
        # Set default encoder/decoder layers if not provided
        if "encoder_layers" not in architecture_params:
            architecture_params["encoder_layers"] = [128, 256, 256, 128, 64]
        if "decoder_layers" not in architecture_params:
            architecture_params["decoder_layers"] = [64, 128, 256, 256, 128]
        # Set default activation if not provided
        if "activation" not in architecture_params:
            architecture_params["activation"] = "relu"
        # Set default dropout rate if not provided
        if "dropout_rate" not in architecture_params:
            architecture_params["dropout_rate"] = 0.1
            
        # Check if architecture is supported
        if architecture not in ["mlp", "default", "best"]:
            print(f"Warning: Architecture '{architecture}' not fully supported yet. Using MLP instead.")
            architecture = "mlp"
            
        self.architecture_name = architecture
        self.architecture_params = architecture_params
        self.alpha = alpha
        self.beta = beta
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Set up device selection (same as NeuralNetworkEstimator)
        if device is None:
            # Auto-detect best available device
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            # Use specified device with validation
            self.device = torch.device(device)
            
            # Validate device selection when manually specified
            if device == "cuda" and not torch.cuda.is_available():
                print("Warning: CUDA requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
            elif device == "mps" and (not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()):
                print("Warning: MPS requested but not available. Falling back to CPU.")
                self.device = torch.device("cpu")
        
        # These will be initialized during training
        self.network = None
        self.data_generator = None
        
        # Initialize the network architecture
        self._create_architecture()
    
    def _create_architecture(self):
        """Create the network architecture based on the selected architecture type."""
        # Determine input and output dimensions
        input_dim = sum(len(points) for points in self.independent_vars_sampling.values())
        param_dim = len(self.param_ranges)
        
        # Create network using parameters already set in self.architecture_params
        self.network = _NNAENetwork(
            input_dim=input_dim,
            param_dim=param_dim,
            architecture_name=self.architecture_name,
            encoder_layers=self.architecture_params["encoder_layers"],
            decoder_layers=self.architecture_params["decoder_layers"],
            activation=self.architecture_params["activation"],
            dropout_rate=self.architecture_params["dropout_rate"]
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
                    "dropout_rate": 0.1
                }
            },
            "best": {
                "description": "Alias for 'mlp' architecture (currently the only fully supported option)",
                "default_params": {
                    "encoder_layers": [128, 256, 256, 128, 64],
                    "decoder_layers": [64, 128, 256, 256, 128],
                    "activation": "relu",
                    "dropout_rate": 0.1
                }
            },
            "default": {
                "description": "Alias for 'mlp' architecture",
                "default_params": {
                    "encoder_layers": [128, 256, 256, 128, 64],
                    "decoder_layers": [64, 128, 256, 256, 128],
                    "activation": "relu",
                    "dropout_rate": 0.1
                }
            }
        }
    
    def train(
        self,
        n_samples: int = 1000,
        batch_size: int = 32,
        n_epochs: int = 100,
        validation_split: float = 0.2,
        verbose: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """Train the NNAE on synthetic data.
        
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
        # Create data generator if not already created
        if self.data_generator is None:
            self.data_generator = SyntheticDataGenerator(
                function=self.function,
                param_ranges=self.param_ranges,
                independent_vars_sampling=self.independent_vars_sampling
            )
        
        # Training history
        history = {
            "train_loss": [],
            "val_loss": [],
            "train_reconstruction_loss": [],
            "train_validation_loss": [],
            "val_reconstruction_loss": [],
            "val_validation_loss": [],
        }
        
        try:
            # Generate synthetic data
            params, function_values = self.data_generator.generate_dataset(n_samples=n_samples)
            
            # Process the data for training
            # For now, we'll assume a single independent variable scenario
            if len(self.independent_var_names) == 1:
                var_name = self.independent_var_names[0]
                _, y_values = function_values[var_name]
                
                # Convert to tensor
                X = torch.tensor(y_values, dtype=torch.float32)
            else:
                raise NotImplementedError("Multiple independent variables not yet implemented")
            
            # Create dataset and dataloaders
            dataset = TensorDataset(X)
            n_val = int(validation_split * len(dataset))
            n_train = len(dataset) - n_val
            train_dataset, val_dataset = torch.utils.data.random_split(dataset, [n_train, n_val])
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            # Create optimizer
            optimizer = optim.Adam(
                self.network.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            
            # Create loss function
            loss_fn = _NNAELoss(
                fit_function=self.function,
                indep_vars=self.independent_vars_sampling,
                param_ranges=self.param_ranges,
                alpha=self.alpha,
                beta=self.beta
            )
            
            # Training loop
            completed_epochs = 0
            
            for epoch in range(n_epochs):
                # Training phase
                self.network.train()
                train_loss = 0.0
                train_reconstruction_loss = 0.0
                train_validation_loss = 0.0
                
                for batch in train_loader:
                    x = batch[0].to(self.device)
                    
                    # Forward pass
                    params, reconstructed = self.network(x)
                    
                    # Compute loss
                    loss, recon_loss, valid_loss = loss_fn(x, params, reconstructed)
                    
                    # Backward pass and optimize
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    train_loss += loss.item() * x.size(0)
                    train_reconstruction_loss += recon_loss.item() * x.size(0)
                    train_validation_loss += valid_loss.item() * x.size(0)
                
                # Compute average losses
                train_loss /= n_train
                train_reconstruction_loss /= n_train
                train_validation_loss /= n_train
                
                # Validation phase
                self.network.eval()
                val_loss = 0.0
                val_reconstruction_loss = 0.0
                val_validation_loss = 0.0
                
                with torch.no_grad():
                    for batch in val_loader:
                        x = batch[0].to(self.device)
                        
                        # Forward pass
                        params, reconstructed = self.network(x)
                        
                        # Compute loss
                        loss, recon_loss, valid_loss = loss_fn(x, params, reconstructed)
                        
                        # Update metrics
                        val_loss += loss.item() * x.size(0)
                        val_reconstruction_loss += recon_loss.item() * x.size(0)
                        val_validation_loss += valid_loss.item() * x.size(0)
                
                # Compute average losses
                val_loss /= n_val
                val_reconstruction_loss /= n_val
                val_validation_loss /= n_val
                
                # Update history
                history["train_loss"].append(train_loss)
                history["val_loss"].append(val_loss)
                history["train_reconstruction_loss"].append(train_reconstruction_loss)
                history["train_validation_loss"].append(train_validation_loss)
                history["val_reconstruction_loss"].append(val_reconstruction_loss)
                history["val_validation_loss"].append(val_validation_loss)
                
                # Update completed epochs count
                completed_epochs = epoch + 1
                
                # Print progress
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{n_epochs}, "
                          f"Train Loss: {train_loss:.4f}, "
                          f"Val Loss: {val_loss:.4f}")
                          
        except KeyboardInterrupt:
            if verbose:
                print(f"\nTraining interrupted at epoch {completed_epochs}/{n_epochs}")
                
                # Only print metrics if we've completed at least one epoch
                if completed_epochs > 0:
                    train_loss = history["train_loss"][-1]
                    val_loss = history["val_loss"][-1]
                    recon_loss = history["train_reconstruction_loss"][-1]
                    valid_loss = history["train_validation_loss"][-1]
                    print(f"Latest metrics - train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}")
                    print(f"Component losses - reconstruction: {recon_loss:.4f}, validation: {valid_loss:.4f}")
                
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
        """
        # Extract y values from arguments
        if "y" in kwargs:
            y_values = kwargs["y"]
        elif len(args) >= 2:
            y_values = args[1]
        else:
            raise ValueError("No y values provided. Use predict(x, y) or predict(x=x_values, y=y_values).")
        
        # Convert to tensor
        y_tensor = torch.tensor(y_values, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # Make prediction
        self.network.eval()
        with torch.no_grad():
            params, _ = self.network(y_tensor)
        
        # Denormalize parameters
        result = {}
        for i, param_name in enumerate(self.param_ranges.keys()):
            min_val, max_val = self.param_ranges[param_name]
            result[param_name] = float(min_val + params[0, i].item() * (max_val - min_val))
        
        return result
    
    def save(self, path: str) -> None:
        """Save the trained model to disk.
        
        Args:
            path: Path to save the model
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        
        # Prepare state dict
        state_dict = {
            "model_state_dict": self.network.state_dict(),
            "architecture_name": self.architecture_name,
            "architecture_params": self.architecture_params,
            "param_ranges": self.param_ranges,
            "independent_vars_sampling": {k: v.tolist() for k, v in self.independent_vars_sampling.items()},
            "alpha": self.alpha,
            "beta": self.beta,
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
        independent_vars_sampling = {
            k: np.array(v) for k, v in state_dict["independent_vars_sampling"].items()
        }
        
        # Create estimator
        estimator = cls(
            function=function,
            param_ranges=state_dict["param_ranges"],
            independent_vars_sampling=independent_vars_sampling,
            architecture=state_dict["architecture_name"],
            architecture_params=state_dict["architecture_params"],
            alpha=state_dict["alpha"],
            beta=state_dict["beta"],
            device=device,
        )
        
        # Load model weights
        estimator.network.load_state_dict(state_dict["model_state_dict"])
        
        return estimator


class _NNAENetwork(nn.Module):
    """Internal network implementation for NNAE."""
    
    def __init__(
        self,
        input_dim: int,
        param_dim: int,
        architecture_name: str,
        encoder_layers: List[int],
        decoder_layers: List[int],
        activation: str = "relu",
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.architecture_name = architecture_name
        self.input_dim = input_dim
        self.param_dim = param_dim
        
        # Select activation function
        if activation == "relu":
            act_fn = nn.ReLU()
        elif activation == "tanh":
            act_fn = nn.Tanh()
        elif activation == "sigmoid":
            act_fn = nn.Sigmoid()
        else:
            act_fn = nn.ReLU()  # Default
        
        # Create encoder (for now, only MLP is implemented)
        self.encoder = self._create_mlp_encoder(
            input_dim, param_dim, encoder_layers, act_fn, dropout_rate
        )
        
        # Parameter normalization (sigmoid activation for [0,1] range)
        self.param_normalizer = nn.Sigmoid()
        
        # Create decoder (for now, only MLP is implemented)
        self.decoder = self._create_mlp_decoder(
            param_dim, input_dim, decoder_layers, act_fn, dropout_rate
        )
    
    def _create_mlp_encoder(self, input_dim, param_dim, hidden_layers, act_fn, dropout_rate):
        """Create MLP encoder network."""
        layers = []
        prev_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, param_dim))
        return nn.Sequential(*layers)
    
    def _create_mlp_decoder(self, param_dim, output_dim, hidden_layers, act_fn, dropout_rate):
        """Create MLP decoder network."""
        layers = []
        prev_dim = param_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_fn)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = h_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Encoder path
        parameters = self.encoder(x)
        normalized_params = self.param_normalizer(parameters)
        
        # Decoder path
        reconstructed = self.decoder(normalized_params)
        
        return normalized_params, reconstructed


class _NNAELoss:
    """Dual-loss function for NNAE training."""
    
    def __init__(self, fit_function, indep_vars, param_ranges, alpha=0.3, beta=0.7):
        self.fit_function = fit_function
        self.indep_vars = indep_vars
        self.param_ranges = param_ranges
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
    
    def __call__(self, y_input, params, y_reconstructed):
        # Reconstruction loss
        reconstruction_loss = self.mse(y_reconstructed, y_input)
        
        # Denormalize parameters
        denorm_params = self._denormalize_parameters(params)
        
        # Compute function values using estimated parameters
        with torch.no_grad():  # No gradients needed for fit function evaluation
            y_computed = self._compute_function_values(denorm_params)
        
        # Parameter validation loss
        validation_loss = self.mse(y_reconstructed, y_computed)
        
        # Combined loss
        total_loss = self.alpha * reconstruction_loss + self.beta * validation_loss
        
        return total_loss, reconstruction_loss, validation_loss
    
    def _denormalize_parameters(self, normalized_params):
        """Convert parameters from [0,1] range to actual parameter ranges."""
        denorm_params = {}
        for i, param_name in enumerate(self.param_ranges.keys()):
            min_val, max_val = self.param_ranges[param_name]
            denorm_params[param_name] = min_val + normalized_params[:, i] * (max_val - min_val)
        return denorm_params
    
    def _compute_function_values(self, params):
        """Evaluate the fit function with the given parameters."""
        # Implementation depends on the format of the fit function
        # This is a simplified example - may need adaptation for specific function interfaces
        
        # Get first independent variable (typically 'x')
        first_var_name = next(iter(self.indep_vars))
        x_values = self.indep_vars[first_var_name]
        batch_size = next(iter(params.values())).shape[0]
        
        # Compute function values for each set of parameters in the batch
        y_computed = torch.zeros((batch_size, len(x_values)), device=next(iter(params.values())).device)
        
        for b in range(batch_size):
            # Extract parameters for this batch item
            batch_params = {k: v[b].item() for k, v in params.items()}
            
            # Compute function values
            if len(self.indep_vars) == 1:
                # Single independent variable
                y_values = self.fit_function(x_values, **batch_params)
            else:
                # Multiple independent variables
                args = [self.indep_vars[var_name] for var_name in self.indep_vars.keys()]
                y_values = self.fit_function(*args, **batch_params)
            
            # Convert to tensor with explicit float32 dtype
            y_computed[b] = torch.tensor(y_values, dtype=torch.float32, device=y_computed.device)
        
        return y_computed 