"""End-to-end test for ZeroGuess using a Gaussian function."""

import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
import os
import datetime
import pathlib

# Import test utilities
from ..conftest import set_random_seeds, gaussian_function, sample_data_1d
from ..test_utils import calculate_parameter_error, is_within_tolerance, calculate_curve_fit_quality

# Import the real ZeroGuess components
from zeroguess.estimators.nn_estimator import NeuralNetworkEstimator
from zeroguess.utils.visualization import plot_fit_comparison, plot_parameter_comparison, plot_training_history


# Define the output directory for visualization files
VISUALIZATION_OUTPUT_DIR = os.path.join("tests", "output", "visualizations")


def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_output_dir_exists():
    """Create the visualization output directory if it doesn't exist."""
    pathlib.Path(VISUALIZATION_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


class TestGaussianWorkflow:
    """End-to-end tests for the full ZeroGuess workflow with a Gaussian function."""
    
    def test_full_workflow(self, set_random_seeds, gaussian_function, sample_data_1d):
        """Test the full ZeroGuess workflow from estimation to evaluate prediction quality."""
        # Get sample data
        x_data, y_data, true_params = sample_data_1d
        
        # Step 1: Define parameter ranges for Gaussian
        param_ranges = {
            'amplitude': (0, 10),
            'center': (-5, 5),
            'width': (0.1, 2)
        }
        
        # Step 2: Define sampling points
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 50)  # Match the x_data dimension from sample_data_1d
        }
        
        # Step 3: Create real estimator with a smaller network for faster tests
        estimator = NeuralNetworkEstimator(
            function=gaussian_function, 
            param_ranges=param_ranges, 
            independent_vars_sampling=independent_vars_sampling,
            hidden_layers=[16, 32, 16],  # Smaller network for faster training
            learning_rate=0.01  # Faster learning rate for tests
        )
        
        # Step 4: Train the estimator with reduced samples and epochs for testing
        training_metrics = estimator.train(
            n_samples=200,  # Reduced sample size for faster tests
            epochs=20,      # Fewer epochs for faster tests
            batch_size=32,
            add_noise=True,
            noise_level=0.1
        )
        assert estimator.is_trained
        
        # Step 5: Predict parameters for experimental data
        predicted_params = estimator.predict(x_data, y_data)
        
        # Step 6: Verify predicted parameters are reasonable
        assert set(predicted_params.keys()) == set(param_ranges.keys())
        
        # Calculate errors compared to true parameters
        errors = calculate_parameter_error(predicted_params, true_params)
        
        # Check if errors are within tolerance (using a higher tolerance since this is real ML)
        for param_name, error in errors.items():
            assert error <= 2.0, f"Error for {param_name} too high: {error:.2f}"
        
        # Step 7: Evaluate prediction quality
        # Use the predicted parameters to generate curve values
        y_predicted = gaussian_function(x_data, **predicted_params)
        
        # Calculate RMSE between predicted curve and actual data
        residuals = y_data - y_predicted
        rmse = np.sqrt(np.mean(residuals**2))
        
        # RMSE should be reasonably small for a good prediction
        # Using a higher threshold since we're using real ML with limited training
        assert rmse < 5.0, f"RMSE too high: {rmse:.2f}"
        
        # Step 8: Verify the predicted parameters can reproduce the original signal
        # Generate clean data with predicted parameters
        y_clean_predicted = gaussian_function(x_data, **predicted_params)
        
        # Generate clean data with true parameters
        y_clean_true = gaussian_function(x_data, **true_params)
        
        # Calculate correlation between predicted and true clean signals
        correlation = np.corrcoef(y_clean_predicted, y_clean_true)[0, 1]
        
        # Correlation should be high
        assert correlation > 0.5, f"Correlation too low: {correlation:.2f}"
    
    def test_estimator_performance_benchmark(self, benchmark, set_random_seeds, gaussian_function, sample_data_1d):
        """Benchmark test for measuring the performance of the neural network estimator."""
        # Get sample data
        x_data, y_data, true_params = sample_data_1d
        
        # Define parameter ranges and sampling points
        param_ranges = {
            'amplitude': (0, 10),
            'center': (-5, 5),
            'width': (0.1, 2)
        }
        
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 50)
        }
        
        # Create estimator with tiny network for benchmark
        estimator = NeuralNetworkEstimator(
            function=gaussian_function, 
            param_ranges=param_ranges, 
            independent_vars_sampling=independent_vars_sampling,
            hidden_layers=[8, 16, 8],  # Very small network for benchmarking
            learning_rate=0.01
        )
        
        # Define the benchmark function that includes training and prediction
        def run_workflow():
            # Train with minimal epochs and samples for benchmarking
            estimator.train(
                n_samples=50,  # Very small dataset for benchmarking
                epochs=5,      # Minimal epochs for benchmarking
                batch_size=16,
                add_noise=True,
                noise_level=0.1
            )
            
            # Predict parameters
            return estimator.predict(x_data, y_data)
        
        # Run the benchmark
        result = benchmark(run_workflow)
        
        # Verify that the results are still reasonable
        predicted_params = result
        
        # Basic sanity check that predicted parameters are within the specified ranges
        for param_name, (min_val, max_val) in param_ranges.items():
            assert min_val <= predicted_params[param_name] <= max_val, f"Parameter {param_name} outside expected range"

    def test_visualization_functions(self, set_random_seeds, gaussian_function, sample_data_1d, monkeypatch):
        """Test the visualization functions using the end-to-end workflow components."""
        # Mock the plt.show() to avoid displaying figures during tests
        monkeypatch.setattr(plt, 'show', lambda: None)
        
        # Ensure the output directory exists
        ensure_output_dir_exists()
        
        # Generate a timestamp for unique filenames
        timestamp = get_timestamp()
        test_name = "test_gaussian_workflow"
        
        # Get sample data
        x_data, y_data, true_params = sample_data_1d
        
        # Step 1: Define parameter ranges for Gaussian
        param_ranges = {
            'amplitude': (0, 10),
            'center': (-5, 5),
            'width': (0.1, 2)
        }
        
        # Step 2: Define sampling points
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 50)
        }
        
        # Step 3: Create a real estimator with a tiny network for testing
        estimator = NeuralNetworkEstimator(
            function=gaussian_function, 
            param_ranges=param_ranges, 
            independent_vars_sampling=independent_vars_sampling,
            hidden_layers=[8, 16, 8],  # Very small network for testing
            learning_rate=0.01
        )
        
        # Step 4: Train the estimator with minimal data for testing
        training_history = estimator.train(
            n_samples=100,
            epochs=10,
            batch_size=16,
            add_noise=True,
            noise_level=0.1,
            validation_split=0.2,  # Include validation to test plotting training history
            return_history=True  # Get training history for visualization
        )
        
        # Step 5: Predict parameters
        estimated_params = estimator.predict(x_data, y_data)
        
        # Create some fitted parameters (typically from optimization, but we'll use true params with small noise)
        # This simulates what would happen after further refinement with a curve fitting library
        np.random.seed(42)
        fitted_params = {
            param: value + np.random.normal(0, 0.05) 
            for param, value in true_params.items()
        }
        
        # Test 1: Plot fit comparison
        fig1 = plot_fit_comparison(
            function=gaussian_function,
            x_data=x_data,
            y_data=y_data,
            true_params=true_params,
            estimated_params=estimated_params,
            fitted_params=fitted_params
        )
        assert fig1 is not None, "plot_fit_comparison should return a matplotlib figure"
        
        # Save the fit comparison figure
        fit_comparison_path = os.path.join(
            VISUALIZATION_OUTPUT_DIR, 
            f"{test_name}_fit_comparison_{timestamp}.png"
        )
        fig1.savefig(fit_comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved fit comparison visualization to: {fit_comparison_path}")
        
        # Test 2: Plot parameter comparison
        fig2 = plot_parameter_comparison(
            true_params=true_params,
            estimated_params=estimated_params,
            fitted_params=fitted_params
        )
        assert fig2 is not None, "plot_parameter_comparison should return a matplotlib figure"
        
        # Save the parameter comparison figure
        param_comparison_path = os.path.join(
            VISUALIZATION_OUTPUT_DIR, 
            f"{test_name}_parameter_comparison_{timestamp}.png"
        )
        fig2.savefig(param_comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved parameter comparison visualization to: {param_comparison_path}")
        
        # Test 3: Plot training history
        # Extract training and validation losses if available
        train_losses = training_history.get('train_loss', [])
        val_losses = training_history.get('val_loss', [])
        
        # Only test if we have training history
        if train_losses:
            fig3 = plot_training_history(
                train_losses=train_losses,
                val_losses=val_losses
            )
            assert fig3 is not None, "plot_training_history should return a matplotlib figure"
            
            # Save the training history figure
            training_history_path = os.path.join(
                VISUALIZATION_OUTPUT_DIR, 
                f"{test_name}_training_history_{timestamp}.png"
            )
            fig3.savefig(training_history_path, dpi=300, bbox_inches='tight')
            print(f"Saved training history visualization to: {training_history_path}")
        
        # Clean up any open figures
        plt.close('all') 