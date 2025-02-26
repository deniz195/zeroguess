"""End-to-end test for ZeroGuess using a multi-peak Gaussian function with CNN architecture."""

import numpy as np
import pytest
import torch
import matplotlib.pyplot as plt
import os

# Import test utilities
from ..conftest import set_random_seeds, multi_peak_gaussian_function, sample_data_multi_peak
from ..test_utils import calculate_parameter_error, is_within_tolerance, calculate_curve_fit_quality

# Import end-to-end test utilities
from .test_utils import (
    create_and_save_visualizations,
    evaluate_prediction_quality,
    ensure_output_dir_exists,
    get_timestamp,
)

# Import the real ZeroGuess components
from zeroguess.estimators.nn_estimator import NeuralNetworkEstimator
from zeroguess.utils.visualization import plot_fit_comparison, plot_parameter_comparison, plot_training_history


# Define the output directory for visualization files
VISUALIZATION_OUTPUT_DIR = os.path.join("tests", "output", "visualizations")


class TestCNNMultiPeakGaussianWorkflow:
    """End-to-end tests for the full ZeroGuess workflow with a multi-peak Gaussian function using CNN architecture."""
    
    def test_full_workflow(self, set_random_seeds, multi_peak_gaussian_function, sample_data_multi_peak, monkeypatch):
        """Test the full ZeroGuess workflow for multi-peak Gaussian estimation using CNN architecture."""
        # Get sample data
        x_data, y_data, true_params = sample_data_multi_peak
        
        # Step 1: Define parameter ranges for multi-peak Gaussian
        param_ranges = {
            'amp1': (0, 10),
            'center1': (-5, 0),  # First peak in negative region
            'width1': (0.1, 2),
            'amp2': (0, 10),
            'center2': (0, 5),   # Second peak in positive region
            'width2': (0.1, 2)
        }
        
        # Step 2: Define sampling points with higher density for better peak resolution
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 100)  # Match the x_data dimension from sample_data_multi_peak
        }
        
        # Step 3: Create real estimator with CNN architecture optimized for multi-peak patterns
        estimator = NeuralNetworkEstimator(
            function=multi_peak_gaussian_function, 
            param_ranges=param_ranges, 
            independent_vars_sampling=independent_vars_sampling,
            architecture="cnn",  # Specify CNN architecture
            architecture_params={
                "n_conv_layers": 3,
                "filters": [16, 32, 64],
                "kernel_size": 5,
                "pool_size": 2,
                "fc_units": [128, 64],
                "dropout_rate": 0.2
            },
            learning_rate=0.005  # Slightly lower learning rate for stability
        )
        
        # Step 4: Train the estimator with increased samples for the more complex function
        training_history = estimator.train(
            n_samples=400,  # More samples for complex function
            epochs=30,      # More epochs for better convergence
            batch_size=32,
            add_noise=True,
            noise_level=0.15,  # Match the noise level in sample data
            return_history=True  # Return history for visualization
        )
        assert estimator.is_trained
        
        # Step 5: Predict parameters for experimental data
        predicted_params = estimator.predict(x_data, y_data)
        
        # Step 6: Verify predicted parameters are reasonable
        assert set(predicted_params.keys()) == set(param_ranges.keys())
        
        # Using the utility function for evaluation with higher thresholds for multi-peak
        evaluate_prediction_quality(
            function=multi_peak_gaussian_function,
            x_data=x_data,
            y_data=y_data,
            true_params=true_params,
            predicted_params=predicted_params,
            max_param_error=3.0,  # Higher tolerance for multi-peak
            max_rmse=7.0,         # Higher RMSE threshold for multi-peak
            min_correlation=0.5
        )
        
        # Generate and save visualizations
        create_and_save_visualizations(
            function=multi_peak_gaussian_function,
            x_data=x_data,
            y_data=y_data,
            true_params=true_params,
            estimated_params=predicted_params,
            test_name="test_cnn_multi_peak_gaussian_full_workflow",
            monkeypatch=monkeypatch
        )
        
        # Additionally, create and save training history visualization if available
        train_losses = training_history.get('train_loss', [])
        val_losses = training_history.get('val_loss', [])
        
        if train_losses:
            # Ensure the output directory exists
            ensure_output_dir_exists()
            
            # Generate timestamp for unique filename
            timestamp = get_timestamp()
            
            fig3 = plot_training_history(
                train_losses=train_losses,
                val_losses=val_losses
            )
            
            # Save the training history figure
            training_history_path = os.path.join(
                VISUALIZATION_OUTPUT_DIR, 
                f"test_cnn_multi_peak_gaussian_full_workflow_training_history_{timestamp}.png"
            )
            fig3.savefig(training_history_path, dpi=300, bbox_inches='tight')
            print(f"Saved training history visualization to: {training_history_path}")
            
            # Close the figure
            plt.close(fig3)
    
    def test_peak_separation_sensitivity(self, set_random_seeds, multi_peak_gaussian_function, monkeypatch):
        """Test the ability of CNN architecture to resolve peaks with varying degrees of separation."""
        # Define base parameter ranges
        param_ranges = {
            'amp1': (0, 10),
            'center1': (-5, 5),
            'width1': (0.1, 2),
            'amp2': (0, 10),
            'center2': (-5, 5),
            'width2': (0.1, 2)
        }
        
        # Define sampling points
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 100)
        }
        
        # Create a CNN network for this challenging task
        estimator = NeuralNetworkEstimator(
            function=multi_peak_gaussian_function, 
            param_ranges=param_ranges, 
            independent_vars_sampling=independent_vars_sampling,
            architecture="cnn",  # Specify CNN architecture
            architecture_params={
                "n_conv_layers": 4,  # More layers for better pattern recognition
                "filters": [16, 32, 64, 128],  # More filters at deeper layers
                "kernel_size": 5,
                "pool_size": 2,
                "fc_units": [128, 64],
                "dropout_rate": 0.2
            },
            learning_rate=0.005
        )
        
        # Train the estimator with increased samples
        training_history = estimator.train(
            n_samples=400,
            epochs=30,
            batch_size=32,
            add_noise=True,
            noise_level=0.1,
            return_history=True  # Return history for visualization
        )
        assert estimator.is_trained
        
        # Test different peak separation scenarios
        separation_scenarios = [
            # Well-separated peaks (easy case)
            {
                'amp1': 5.0, 'center1': -3.0, 'width1': 0.8,
                'amp2': 3.0, 'center2': 3.0, 'width2': 0.8
            },
            # Moderately separated peaks
            {
                'amp1': 5.0, 'center1': -1.5, 'width1': 0.8,
                'amp2': 3.0, 'center2': 1.5, 'width2': 0.8
            },
            # Closely spaced peaks
            {
                'amp1': 5.0, 'center1': -1.0, 'width1': 0.8,
                'amp2': 3.0, 'center2': 1.0, 'width2': 0.8
            }
        ]
        
        for i, true_params in enumerate(separation_scenarios):
            # Generate test data
            x_data = np.linspace(-10, 10, 100)
            y_clean = multi_peak_gaussian_function(x_data, **true_params)
            
            # Add noise
            np.random.seed(42 + i)  # Different seed for each scenario
            noise = np.random.normal(0, 0.15, size=len(x_data))
            y_data = y_clean + noise
            
            # Predict parameters
            predicted_params = estimator.predict(x_data, y_data)
            
            # Calculate errors
            errors = calculate_parameter_error(predicted_params, true_params)
            
            # Calculate fit quality
            y_predicted = multi_peak_gaussian_function(x_data, **predicted_params)
            quality = calculate_curve_fit_quality(multi_peak_gaussian_function, x_data, y_data, predicted_params)
            
            # Print results for debugging
            print(f"\nScenario {i+1} (Peak Separation: {abs(true_params['center1'] - true_params['center2']):.1f}):")
            for param, error in errors.items():
                print(f"  {param}: true={true_params[param]:.2f}, pred={predicted_params[param]:.2f}, error={error:.2f}")
            print(f"  Fit quality (RMSE): {quality:.4f}")
            
            # Define error thresholds based on separation difficulty
            # Higher tolerance for more difficult cases (closer peaks)
            if i == 0:  # Well-separated peaks
                max_error = 2.0
                max_rmse = 2.0
            elif i == 1:  # Moderately separated
                max_error = 3.0
                max_rmse = 2.5
            else:  # Closely spaced
                max_error = 4.0
                max_rmse = 3.0
                
            # Check individual parameter errors
            for param_name, error in errors.items():
                assert error <= max_error, f"Error for {param_name} too high in scenario {i+1}: {error:.2f}"
            
            # Check overall fit quality
            assert quality <= max_rmse, f"RMSE too high in scenario {i+1}: {quality:.4f}"
            
            # Generate and save visualizations for each scenario
            scenario_name = f"test_cnn_multi_peak_gaussian_separation_{abs(true_params['center1'] - true_params['center2']):.1f}"
            create_and_save_visualizations(
                function=multi_peak_gaussian_function,
                x_data=x_data,
                y_data=y_data,
                true_params=true_params,
                estimated_params=predicted_params,
                test_name=scenario_name,
                monkeypatch=monkeypatch
            )
        
        # Additionally, create and save training history visualization if available
        train_losses = training_history.get('train_loss', [])
        val_losses = training_history.get('val_loss', [])
        
        if train_losses:
            # Ensure the output directory exists
            ensure_output_dir_exists()
            
            # Generate timestamp for unique filename
            timestamp = get_timestamp()
            
            fig3 = plot_training_history(
                train_losses=train_losses,
                val_losses=val_losses
            )
            
            # Save the training history figure
            training_history_path = os.path.join(
                VISUALIZATION_OUTPUT_DIR, 
                f"test_cnn_multi_peak_gaussian_separation_training_history_{timestamp}.png"
            )
            fig3.savefig(training_history_path, dpi=300, bbox_inches='tight')
            print(f"Saved training history visualization to: {training_history_path}")
            
            # Close the figure
            plt.close(fig3)
    
    def test_visualization_functions(self, set_random_seeds, multi_peak_gaussian_function, 
                                    sample_data_multi_peak, monkeypatch):
        """Test the visualization functions using the CNN architecture with multi-peak Gaussian workflow."""
        # Get sample data
        x_data, y_data, true_params = sample_data_multi_peak
        
        # Step 1: Define parameter ranges for multi-peak Gaussian
        param_ranges = {
            'amp1': (0, 10),
            'center1': (-5, 0),
            'width1': (0.1, 2),
            'amp2': (0, 10),
            'center2': (0, 5),
            'width2': (0.1, 2)
        }
        
        # Step 2: Define sampling points
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 100)
        }
        
        # Step 3: Create a real estimator with CNN architecture for testing
        estimator = NeuralNetworkEstimator(
            function=multi_peak_gaussian_function, 
            param_ranges=param_ranges, 
            independent_vars_sampling=independent_vars_sampling,
            architecture="cnn",
            architecture_params={
                "n_conv_layers": 2,  # Smaller network for faster testing
                "filters": [16, 32],
                "kernel_size": 3,
                "pool_size": 2,
                "fc_units": [64, 32],
                "dropout_rate": 0.1
            },
            learning_rate=0.01
        )
        
        # Step 4: Train the estimator with minimal data for testing
        training_history = estimator.train(
            n_samples=200,
            epochs=15,
            batch_size=32,
            add_noise=True,
            noise_level=0.15,
            validation_split=0.2,
            return_history=True
        )
        
        # Step 5: Predict parameters
        estimated_params = estimator.predict(x_data, y_data)
        
        # Test visualization functions using our utility
        fig1, fig2, _ = create_and_save_visualizations(
            function=multi_peak_gaussian_function,
            x_data=x_data,
            y_data=y_data,
            true_params=true_params,
            estimated_params=estimated_params,
            test_name="test_cnn_multi_peak_gaussian_workflow",
            monkeypatch=monkeypatch
        )
        
        # Additionally, test the training history visualization if available
        train_losses = training_history.get('train_loss', [])
        val_losses = training_history.get('val_loss', [])
        
        if train_losses:
            # Ensure the output directory exists
            ensure_output_dir_exists()
            
            # Generate timestamp for unique filename
            timestamp = get_timestamp()
            
            fig3 = plot_training_history(
                train_losses=train_losses,
                val_losses=val_losses
            )
            assert fig3 is not None, "plot_training_history should return a matplotlib figure"
            
            # Save the training history figure
            training_history_path = os.path.join(
                VISUALIZATION_OUTPUT_DIR, 
                f"test_cnn_multi_peak_gaussian_workflow_training_history_{timestamp}.png"
            )
            fig3.savefig(training_history_path, dpi=300, bbox_inches='tight')
            print(f"Saved training history visualization to: {training_history_path}")
            
            # Close the figure
            plt.close(fig3)
        
        # Clean up any remaining open figures
        plt.close('all')
    
    def test_cnn_vs_mlp_comparison(self, set_random_seeds, multi_peak_gaussian_function, sample_data_multi_peak, monkeypatch):
        """Compare CNN architecture performance against MLP for multi-peak Gaussian estimation."""
        # Get sample data
        x_data, y_data, true_params = sample_data_multi_peak
        
        # Define parameter ranges for multi-peak Gaussian
        param_ranges = {
            'amp1': (0, 10),
            'center1': (-5, 0),  # First peak in negative region
            'width1': (0.1, 2),
            'amp2': (0, 10),
            'center2': (0, 5),   # Second peak in positive region
            'width2': (0.1, 2)
        }
        
        # Define sampling points
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 100)
        }
        
        # Create estimators with CNN and MLP architectures
        cnn_estimator = NeuralNetworkEstimator(
            function=multi_peak_gaussian_function, 
            param_ranges=param_ranges, 
            independent_vars_sampling=independent_vars_sampling,
            architecture="cnn",
            architecture_params={
                "n_conv_layers": 3,
                "filters": [16, 32, 64],
                "kernel_size": 5,
                "pool_size": 2,
                "fc_units": [128, 64],
                "dropout_rate": 0.2
            },
            learning_rate=0.005
        )
        
        mlp_estimator = NeuralNetworkEstimator(
            function=multi_peak_gaussian_function, 
            param_ranges=param_ranges, 
            independent_vars_sampling=independent_vars_sampling,
            architecture="mlp",
            architecture_params={
                "hidden_layers": [32, 64, 64, 32],
                "activation": "relu",
                "dropout_rate": 0.2,
                "use_batch_norm": True,
                "use_residual": False
            },
            learning_rate=0.005
        )
        
        # Train both estimators with identical training parameters
        print("\nTraining CNN architecture...")
        cnn_history = cnn_estimator.train(
            n_samples=300,
            epochs=25,
            batch_size=32,
            add_noise=True,
            noise_level=0.15,
            return_history=True
        )
        
        print("\nTraining MLP architecture...")
        mlp_history = mlp_estimator.train(
            n_samples=300,
            epochs=25,
            batch_size=32,
            add_noise=True,
            noise_level=0.15,
            return_history=True
        )
        
        # Predict parameters with both architectures
        cnn_predicted_params = cnn_estimator.predict(x_data, y_data)
        mlp_predicted_params = mlp_estimator.predict(x_data, y_data)
        
        # Calculate errors for both architectures
        cnn_errors = calculate_parameter_error(cnn_predicted_params, true_params)
        mlp_errors = calculate_parameter_error(mlp_predicted_params, true_params)
        
        # Calculate fit quality
        cnn_quality = calculate_curve_fit_quality(multi_peak_gaussian_function, x_data, y_data, cnn_predicted_params)
        mlp_quality = calculate_curve_fit_quality(multi_peak_gaussian_function, x_data, y_data, mlp_predicted_params)
        
        # Print comparison results
        print("\nArchitecture performance comparison:")
        print("Parameter errors (%):")
        for param in true_params.keys():
            print(f"  {param}: CNN={cnn_errors[param]:.2f}%, MLP={mlp_errors[param]:.2f}%")
        
        print(f"Fit quality (RMSE): CNN={cnn_quality:.4f}, MLP={mlp_quality:.4f}")
        
        # Calculate mean error for both architectures
        cnn_mean_error = sum(cnn_errors.values()) / len(cnn_errors)
        mlp_mean_error = sum(mlp_errors.values()) / len(mlp_errors)
        print(f"Mean parameter error: CNN={cnn_mean_error:.2f}%, MLP={mlp_mean_error:.2f}%")
        
        # Generate visualization comparing both architectures
        ensure_output_dir_exists()
        timestamp = get_timestamp()
        
        # Plot comparison of fit
        plt.figure(figsize=(12, 6))
        plt.plot(x_data, y_data, 'k.', alpha=0.3, label='Noisy data')
        plt.plot(x_data, multi_peak_gaussian_function(x_data, **true_params), 'g-', label='True function')
        plt.plot(x_data, multi_peak_gaussian_function(x_data, **cnn_predicted_params), 'r-', label='CNN prediction')
        plt.plot(x_data, multi_peak_gaussian_function(x_data, **mlp_predicted_params), 'b--', label='MLP prediction')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('CNN vs MLP Architecture Comparison for Multi-Peak Gaussian')
        plt.legend()
        plt.grid(True)
        
        # Save the comparison figure
        comparison_path = os.path.join(
            VISUALIZATION_OUTPUT_DIR, 
            f"test_cnn_vs_mlp_comparison_{timestamp}.png"
        )
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"Saved architecture comparison visualization to: {comparison_path}")
        plt.close()
        
        # Plot training history comparison
        plt.figure(figsize=(12, 6))
        plt.plot(cnn_history['train_loss'], 'r-', label='CNN Training Loss')
        plt.plot(cnn_history['val_loss'], 'r--', label='CNN Validation Loss')
        plt.plot(mlp_history['train_loss'], 'b-', label='MLP Training Loss')
        plt.plot(mlp_history['val_loss'], 'b--', label='MLP Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training History Comparison: CNN vs MLP')
        plt.legend()
        plt.grid(True)
        
        # Save the history comparison figure
        history_path = os.path.join(
            VISUALIZATION_OUTPUT_DIR, 
            f"test_cnn_vs_mlp_history_comparison_{timestamp}.png"
        )
        plt.savefig(history_path, dpi=300, bbox_inches='tight')
        print(f"Saved training history comparison to: {history_path}")
        plt.close()
    
    def test_estimator_performance_benchmark(self, benchmark, set_random_seeds, 
                                            multi_peak_gaussian_function, sample_data_multi_peak, monkeypatch):
        """Benchmark test for measuring the performance of CNN architecture with multi-peak functions."""
        # Get sample data
        x_data, y_data, true_params = sample_data_multi_peak
        
        # Define parameter ranges and sampling points
        param_ranges = {
            'amp1': (0, 10),
            'center1': (-5, 0),
            'width1': (0.1, 2),
            'amp2': (0, 10),
            'center2': (0, 5),
            'width2': (0.1, 2)
        }
        
        independent_vars_sampling = {
            'x': np.linspace(-10, 10, 100)
        }
        
        # Create estimator with tiny network for benchmark
        estimator = NeuralNetworkEstimator(
            function=multi_peak_gaussian_function, 
            param_ranges=param_ranges, 
            independent_vars_sampling=independent_vars_sampling,
            architecture="cnn",
            architecture_params={
                "n_conv_layers": 2,  # Small network for benchmarking
                "filters": [8, 16],
                "kernel_size": 3,
                "pool_size": 2,
                "fc_units": [32, 16],
                "dropout_rate": 0.1
            },
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
                noise_level=0.15
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
        
        # Generate and save visualizations for the benchmark results
        create_and_save_visualizations(
            function=multi_peak_gaussian_function,
            x_data=x_data,
            y_data=y_data,
            true_params=true_params,
            estimated_params=predicted_params,
            test_name="test_cnn_multi_peak_gaussian_benchmark",
            monkeypatch=monkeypatch
        ) 