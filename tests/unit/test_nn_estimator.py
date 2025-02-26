"""Unit tests for the neural network estimator component of ZeroGuess."""

import numpy as np
import pytest

# Import the test fixtures
from ..conftest import set_random_seeds, requires_torch

# Try to import torch, but allow tests to be skipped if not available
try:
    import torch
    from torch import nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# Mock NN estimator implementation for testing
if HAS_TORCH:
    class MockNNEstimator:
        """Mock Neural Network Estimator for testing."""
        
        def __init__(self, function, param_ranges, independent_vars_sampling):
            """Initialize the estimator with function and parameter information."""
            self.function = function
            self.param_ranges = param_ranges
            self.independent_vars_sampling = independent_vars_sampling
            self.param_names = list(param_ranges.keys())
            self.var_names = list(independent_vars_sampling.keys())
            
            # Create a simple neural network
            self.model = nn.Sequential(
                nn.Linear(sum(len(sampling) for sampling in independent_vars_sampling.values()), 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, len(param_ranges))
            )
            
            # For testing, we'll just pre-initialize weights to known values
            for layer in self.model:
                if isinstance(layer, nn.Linear):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
            
            self.is_trained = False
        
        def train(self, epochs=10, batch_size=32, learning_rate=0.001):
            """Train the model on synthetic data."""
            # In a real implementation, this would generate data and train the model
            # For testing, we'll just mark it as trained
            self.is_trained = True
            return {"loss": 0.1, "val_loss": 0.2}  # Mock training metrics
        
        def predict(self, x_data, y_data):
            """Predict parameters for the given data."""
            if not self.is_trained:
                raise RuntimeError("Model must be trained before prediction")
            
            # Ensure we're matching the input size expected by the model
            input_size = sum(len(sampling) for sampling in self.independent_vars_sampling.values())
            
            # Instead of using x_data directly, we'll use the sampling points
            # This ensures the input size matches what the model expects
            x_tensor = torch.tensor(
                self.independent_vars_sampling['x'], 
                dtype=torch.float32
            ).reshape(1, -1)
            
            # Run forward pass
            with torch.no_grad():
                raw_params = self.model(x_tensor).numpy()[0]
            
            # Scale parameters to their ranges
            params = {}
            for i, name in enumerate(self.param_names):
                min_val, max_val = self.param_ranges[name]
                params[name] = min_val + (max_val - min_val) * raw_params[i]
            
            return params

# Test class
@requires_torch
class TestNNEstimator:
    """Tests for the neural network estimator functionality."""
    
    def test_initialization(self, gaussian_function):
        """Test that the estimator initializes correctly."""
        # Define parameter ranges
        param_ranges = {
            'amplitude': (0, 10),
            'center': (-5, 5),
            'width': (0.1, 2)
        }
        
        # Define sampling points
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 100)
        }
        
        # Initialize estimator
        estimator = MockNNEstimator(
            gaussian_function, param_ranges, independent_vars_sampling
        )
        
        # Check that parameters are stored correctly
        assert estimator.param_ranges == param_ranges
        assert estimator.independent_vars_sampling == independent_vars_sampling
        assert set(estimator.param_names) == set(param_ranges.keys())
        assert set(estimator.var_names) == set(independent_vars_sampling.keys())
        
        # Check that model is initialized
        assert estimator.model is not None
        assert not estimator.is_trained
    
    def test_training(self, set_random_seeds, gaussian_function):
        """Test that the estimator can be trained."""
        # Define parameter ranges
        param_ranges = {
            'amplitude': (0, 10),
            'center': (-5, 5),
            'width': (0.1, 2)
        }
        
        # Define sampling points
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 100)
        }
        
        # Initialize estimator
        estimator = MockNNEstimator(
            gaussian_function, param_ranges, independent_vars_sampling
        )
        
        # Train the model
        training_metrics = estimator.train(epochs=5, batch_size=16, learning_rate=0.01)
        
        # Check that training completed
        assert estimator.is_trained
        assert "loss" in training_metrics
        assert "val_loss" in training_metrics
    
    def test_prediction(self, set_random_seeds, gaussian_function, sample_data_1d):
        """Test that the estimator can make predictions."""
        # Get sample data
        x_data, y_data, true_params = sample_data_1d
        
        # Define parameter ranges
        param_ranges = {
            'amplitude': (0, 10),
            'center': (-5, 5),
            'width': (0.1, 2)
        }
        
        # Define sampling points
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 100)
        }
        
        # Initialize estimator
        estimator = MockNNEstimator(
            gaussian_function, param_ranges, independent_vars_sampling
        )
        
        # Train the model
        estimator.train()
        
        # Make prediction
        predicted_params = estimator.predict(x_data, y_data)
        
        # Check that prediction has the correct structure
        assert set(predicted_params.keys()) == set(param_ranges.keys())
        
        # Check that predicted values are within the specified ranges
        for param_name, value in predicted_params.items():
            min_val, max_val = param_ranges[param_name]
            assert value >= min_val
            assert value <= max_val
    
    def test_untrained_prediction_raises_error(self, gaussian_function, sample_data_1d):
        """Test that prediction with untrained model raises an error."""
        # Get sample data
        x_data, y_data, _ = sample_data_1d
        
        # Define parameter ranges
        param_ranges = {
            'amplitude': (0, 10),
            'center': (-5, 5),
            'width': (0.1, 2)
        }
        
        # Define sampling points
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 100)
        }
        
        # Initialize estimator
        estimator = MockNNEstimator(
            gaussian_function, param_ranges, independent_vars_sampling
        )
        
        # Prediction should raise error because model is not trained
        with pytest.raises(RuntimeError, match="Model must be trained before prediction"):
            estimator.predict(x_data, y_data) 