#!/usr/bin/env python
"""
Example demonstrating ZeroGuess's Neural Network Autoencoder Estimator (NNAE) for parameter estimation with a wavelet function.

This example shows how to use the NNAE estimator for parameter estimation and compares its performance
with the standard neural network estimator and traditional curve fitting.

The true parameters are randomly generated and the same noisy dataset is used for all methods to ensure direct comparison.

Usage:
    python wavelet_nnae_example.py [--random] [--epochs EPOCHS]

Options:
    --random         Use truly random parameters (no fixed seed)
    --epochs EPOCHS  Number of training epochs (default: 1000)
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

import zeroguess
from zeroguess.functions.standard import WaveletFunction
from zeroguess.functions.utils import add_gaussian_noise
from zeroguess.utils.visualization import (
    plot_fit_comparison,
    plot_parameter_comparison,
    plot_training_history,
)

# Add the parent directory to the path so we can import zeroguess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def example_nnae_estimator(true_params, x_data, y_data, x_sampling=None, n_epochs=1000):
    """Example of using the NNAE estimator.

    Args:
        true_params: Dictionary of true parameter values to use
        x_data: Independent variable values for fitting
        y_data: Dependent variable values (noisy data) for fitting
        x_sampling: Optional pre-defined sampling points for training
        n_epochs: Number of training epochs
    """
    print("\n=========================================================")
    print("Running example: NNAE Estimator (Wavelet)")
    print("=========================================================")

    print("Using true parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")

    # Get the wavelet function and its parameter ranges
    wavelet_func = WaveletFunction()
    
    # Create and train NNAE estimator
    print("Creating and training NNAE estimator...")
    nnae_estimator = zeroguess.create_estimator(
        function=wavelet_func,
        param_ranges=wavelet_func.param_ranges,
        independent_vars_sampling={
            "x": x_sampling,
        },
        estimator_type="nnae",  # Specify NNAE estimator
        architecture="mlp",     # Only MLP is supported for NNAE currently
        architecture_params={
            "encoder_layers": [128, 256, 256, 128, 64],
            "decoder_layers": [64, 128, 256, 256, 128],
            # "encoder_layers": [1024, 1024, 512, 256, 128, 64],
            # "decoder_layers": [64, 128, 256, 512, 1024, 1024],
            "activation": "relu",
            "dropout_rate": 0.1,
        },
        alpha=0.5,  # Reconstruction loss weight
        beta=0.5,    # Parameter validation loss weight
        device="cpu",        
    )

    # Train the NNAE estimator
    nnae_training_results = nnae_estimator.train(
        n_samples=1000,
        n_epochs=n_epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=True,
    )

    # Get parameter estimates from NNAE
    print("Predicting parameters using NNAE estimator...")
    nnae_estimated_params = nnae_estimator.predict(x=x_data, y=y_data)
    print("NNAE estimated parameters:")
    for param, value in nnae_estimated_params.items():
        print(f"  {param}: {value:.6f}")

    # For comparison, create and train standard neural network estimator
    print("\nCreating and training standard neural network estimator for comparison...")
    nn_estimator = zeroguess.create_estimator(
        function=wavelet_func,
        param_ranges=wavelet_func.param_ranges,
        independent_vars_sampling={
            "x": x_sampling,
        },
        estimator_type="neural_network",  # Standard neural network estimator
        device="cpu",
    )

    # Train the standard neural network estimator
    nn_training_results = nn_estimator.train(
        n_samples=1000,
        n_epochs=n_epochs,
        batch_size=32,
        validation_split=0.2,
        verbose=True,
    )

    # Get parameter estimates from standard neural network
    print("Predicting parameters using standard neural network estimator...")
    nn_estimated_params = nn_estimator.predict(x_data, y_data)
    print("Standard NN estimated parameters:")
    for param, value in nn_estimated_params.items():
        print(f"  {param}: {value:.6f}")

    # Use NNAE estimated parameters for curve fitting
    print("\nPerforming curve fitting with NNAE estimated parameters...")
    
    # Define bounds for curve fitting to help convergence
    param_ranges = wavelet_func.param_ranges
    bounds = (
        [param_ranges[p][0] for p in ["frequency", "phase", "position", "width"]],
        [param_ranges[p][1] for p in ["frequency", "phase", "position", "width"]]
    )

    popt, _ = optimize.curve_fit(
        wavelet_func,
        x_data,
        y_data,
        p0=[nnae_estimated_params[param] for param in ["frequency", "phase", "position", "width"]],
        bounds=bounds,
    )

    # Convert popt to dictionary
    fitted_params = {
        "frequency": popt[0],
        "phase": popt[1],
        "position": popt[2],
        "width": popt[3],
    }
    print("Fitted parameters (using NNAE estimates as initial guess):")
    for param, value in fitted_params.items():
        print(f"  {param}: {value:.6f}")

    # Plot NNAE training history
    plt.figure(figsize=(10, 6))
    plt.plot(nnae_training_results["train_loss"], label="Train Loss")
    plt.plot(nnae_training_results["val_loss"], label="Validation Loss")
    plt.title("NNAE Training History: Wavelet Parameter Estimation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("wavelet_nnae_training_history.png")
    
    # Plot standard NN training history
    plt.figure(figsize=(10, 6))
    plt.plot(nn_training_results["train_loss"], label="Train Loss")
    plt.plot(nn_training_results["val_loss"], label="Validation Loss")
    plt.title("Standard NN Training History: Wavelet Parameter Estimation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("wavelet_nn_training_history.png")

    # Plot NNAE component losses
    plt.figure(figsize=(10, 6))
    plt.plot(nnae_training_results["train_reconstruction_loss"], label="Reconstruction Loss")
    plt.plot(nnae_training_results["train_validation_loss"], label="Parameter Validation Loss")
    plt.title("NNAE Component Losses: Wavelet Parameter Estimation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("wavelet_nnae_component_losses.png")

    # Plot fit comparison
    y_true = wavelet_func(x_data, **true_params)
    y_nnae = wavelet_func(x_data, **nnae_estimated_params)
    y_nn = wavelet_func(x_data, **nn_estimated_params)
    y_fitted = wavelet_func(x_data, **fitted_params)
    
    plt.figure(figsize=(12, 8))
    plt.plot(x_data, y_data, 'ko', alpha=0.3, label='Noisy Data')
    plt.plot(x_data, y_true, 'r-', linewidth=2, label='True Function')
    plt.plot(x_data, y_nnae, 'g--', linewidth=2, label='NNAE Estimate')
    plt.plot(x_data, y_nn, 'b-.', linewidth=2, label='Standard NN Estimate')
    plt.plot(x_data, y_fitted, 'm:', linewidth=2, label='Fitted (NNAE + curve_fit)')
    plt.title("Wavelet Fit Comparison")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.savefig("wavelet_nnae_fit_comparison.png")

    # Plot parameter comparison
    param_names = list(true_params.keys())
    n_params = len(param_names)
    
    plt.figure(figsize=(12, 8))
    x = np.arange(n_params)
    width = 0.2
    
    # Normalize parameters to [0,1] for better visualization
    normalized_true = []
    normalized_nnae = []
    normalized_nn = []
    normalized_fitted = []
    
    for param in param_names:
        min_val, max_val = nnae_estimator.param_ranges[param]
        range_val = max_val - min_val
        
        normalized_true.append((true_params[param] - min_val) / range_val)
        normalized_nnae.append((nnae_estimated_params[param] - min_val) / range_val)
        normalized_nn.append((nn_estimated_params[param] - min_val) / range_val)
        normalized_fitted.append((fitted_params[param] - min_val) / range_val)
    
    plt.bar(x - 1.5*width, normalized_true, width, label='True')
    plt.bar(x - 0.5*width, normalized_nnae, width, label='NNAE')
    plt.bar(x + 0.5*width, normalized_nn, width, label='Standard NN')
    plt.bar(x + 1.5*width, normalized_fitted, width, label='Fitted')
    
    plt.xlabel('Parameter')
    plt.ylabel('Normalized Value [0,1]')
    plt.title('Parameter Comparison (Normalized)')
    plt.xticks(x, param_names)
    plt.legend()
    plt.grid(True, axis='y')
    plt.savefig("wavelet_nnae_parameter_comparison.png")
    
    # Calculate and print error metrics
    print("\nParameter estimation error (absolute):")
    print("                  NNAE      Standard NN")
    for param in param_names:
        nnae_error = abs(nnae_estimated_params[param] - true_params[param])
        nn_error = abs(nn_estimated_params[param] - true_params[param])
        print(f"{param:10}: {nnae_error:.6f}   {nn_error:.6f}")
    
    # Calculate mean absolute error across all parameters
    nnae_mae = np.mean([abs(nnae_estimated_params[param] - true_params[param]) for param in param_names])
    nn_mae = np.mean([abs(nn_estimated_params[param] - true_params[param]) for param in param_names])
    fitted_mae = np.mean([abs(fitted_params[param] - true_params[param]) for param in param_names])
    
    print(f"\nMean Absolute Error:")
    print(f"NNAE Estimator:          {nnae_mae:.6f}")
    print(f"Standard NN Estimator:   {nn_mae:.6f}")
    print(f"Fitted (NNAE + curve_fit): {fitted_mae:.6f}")
    
    print("\nSaved plots to current directory")
    return x_sampling  # Return sampling points for reuse


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run NNAE estimator example with a wavelet function")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use truly random parameters (no fixed seed)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs (default: 1000)",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility unless --random flag is used
    if not args.random:
        np.random.seed(123)
        print("Using fixed random seed (123) for reproducible results")
    else:
        print("Using truly random parameters for each run")

    # Create wavelet function to get default sampling points and parameter ranges
    wavelet_func = WaveletFunction()
    
    # Generate random parameters using the built-in method
    true_params = wavelet_func.get_random_params()
    print("\nGenerated true parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")

    # Use default sampling points from WaveletFunction
    x_sampling = wavelet_func.default_independent_vars["x"]

    # Generate clean data and add noise using the utility function
    x_data = x_sampling.copy()
    y_clean = wavelet_func(x_data, **true_params)
    y_data = add_gaussian_noise(y_clean, sigma=0.05, relative=True)
    print("\nGenerated noisy data for testing")

    # Run NNAE example
    example_nnae_estimator(true_params, x_data, y_data, x_sampling, n_epochs=args.epochs) 