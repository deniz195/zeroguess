#!/usr/bin/env python
"""
Example demonstrating ZeroGuess for a Gaussian-enveloped cosine wavelet function.

This example shows the use of ZeroGuess for a more complex function with multiple parameters:
- frequency: The frequency of the cosine oscillation
- phase: The phase shift of the cosine
- position: The center position of the Gaussian envelope
- width: The width of the Gaussian envelope

The true parameters are randomly generated within reasonable ranges each time the script is run.

Usage:
    python wavelet_example.py [--random] [--method METHOD]
    
Options:
    --random         Use truly random parameters (no fixed seed)
    --method METHOD  Specify which method(s) to run: 'all', 'basic', 'scipy', or 'lmfit' (default: 'all')
"""
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os

# Check if lmfit is installed
try:
    import lmfit
    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False
    print("Note: lmfit package is not installed. The lmfit example will be skipped.")
    print("To use lmfit integration, install it with: pip install lmfit")

# Add the parent directory to the path so we can import zeroguess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import zeroguess
from zeroguess.integration import scipy_integration
# Only import lmfit_integration if lmfit is available
if LMFIT_AVAILABLE:
    from zeroguess.integration import lmfit_integration
    
from zeroguess.utils.visualization import (
    plot_fit_comparison,
    plot_training_history,
    plot_parameter_comparison,
)


def wavelet(x, frequency, phase, position, width):
    """
    Gaussian-enveloped cosine wavelet function.
    
    Args:
        x: Independent variable
        frequency: Frequency of the cosine oscillation
        phase: Phase shift of the cosine in radians
        position: Center position of the Gaussian envelope
        width: Width of the Gaussian envelope
    
    Returns:
        Wavelet function values
    """
    # Calculate the Gaussian envelope
    envelope = np.exp(-(x - position)**2 / (2 * width**2))
    
    # Calculate the cosine wave
    wave = np.cos(2 * np.pi * frequency * (x - position) + phase)
    
    # Combine envelope and wave
    return envelope * wave


def generate_noisy_data(x, true_params, noise_level=0.05):
    """Generate noisy data for testing.
    
    Args:
        x: Independent variable values
        true_params: Dictionary of true parameter values
        noise_level: Relative noise level as a fraction of the signal amplitude
        
    Returns:
        Noisy data points
    """
    y_true = wavelet(x, **true_params)
    noise = np.random.normal(0, noise_level * np.max(np.abs(y_true)), size=len(x))
    return y_true + noise


def generate_random_true_params():
    """Generate random true parameters within reasonable ranges.
    
    The parameters are randomly chosen within these ranges:
    - frequency: 0.2 to 1.5 oscillations per unit x
      (too high will cause aliasing, too low won't show oscillation)
    - phase: 0 to 2π radians (covers all possible phase shifts)
    - position: -3.0 to 3.0 (keeps the wavelet within the sampling window)
    - width: 1.0 to 3.0 (ensures the wavelet is neither too narrow nor too wide)
    
    Returns:
        Dictionary of randomly generated parameter values
    """
    return {
        'frequency': np.random.uniform(0.2, 1.5),    # oscillations per unit x
        'phase': np.random.uniform(0, 2*np.pi),      # phase shift in radians
        'position': np.random.uniform(-3.0, 3.0),    # center position
        'width': np.random.uniform(1.0, 3.0),        # width of the Gaussian envelope
    }


def example_basic_usage():
    """Example of basic usage of ZeroGuess with a wavelet function."""
    print("\n=========================================================")
    print("Running example: Basic Usage (Wavelet)")
    print("=========================================================")
    
    # Generate random true parameters
    true_params = generate_random_true_params()
    print("Randomly generated true parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Define the sampling points - use the same for training and prediction
    x_sampling = np.linspace(-10, 10, 200)
    
    # Generate data
    x_data = x_sampling.copy()  # Use the same x points
    y_data = generate_noisy_data(x_data, true_params)
    
    # Create and train estimator
    print("Creating and training estimator...")
    estimator = zeroguess.create_estimator(
        function=wavelet,
        param_ranges={
            'frequency': (0.1, 2.0),
            'phase': (0, 2*np.pi),
            'position': (-5.0, 5.0),
            'width': (0.5, 5.0),
        },
        independent_vars_sampling={
            'x': x_sampling,
        },
    )
    
    training_results = estimator.train(
        n_samples=1500,  # More samples for a more complex function
        epochs=100,      # More epochs for better convergence
        batch_size=32,
        add_noise=True,
        noise_level=0.05,
    )
    
    # Get parameter estimates
    print("Predicting parameters...")
    estimated_params = estimator.predict(x_data, y_data)
    print("Estimated parameters:")
    for param, value in estimated_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Use estimated parameters for curve fitting
    print("Performing curve fitting with estimated parameters...")
    popt, _ = optimize.curve_fit(
        wavelet, x_data, y_data,
        p0=[estimated_params[param] for param in ['frequency', 'phase', 'position', 'width']],
    )
    
    # Convert popt to dictionary
    fitted_params = {
        'frequency': popt[0],
        'phase': popt[1],
        'position': popt[2],
        'width': popt[3],
    }
    print("Fitted parameters:")
    for param, value in fitted_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Plot results
    plot_training_history(
        training_results['train_losses'],
        training_results['val_losses'],
        title="Training History: Wavelet Parameter Estimation",
    )
    plt.savefig("wavelet_training_history.png")
    
    plot_fit_comparison(
        wavelet, x_data, y_data,
        true_params=true_params,
        estimated_params=estimated_params,
        fitted_params=fitted_params,
        title="Wavelet Fit Comparison",
    )
    plt.savefig("wavelet_fit_comparison.png")
    
    plot_parameter_comparison(
        true_params, estimated_params, fitted_params,
        title="Wavelet Parameter Comparison",
    )
    plt.savefig("wavelet_parameter_comparison.png")
    
    print("Saved plots to current directory")


def example_scipy_integration():
    """Example of using ZeroGuess with SciPy integration for a wavelet function."""
    print("\n=========================================================")
    print("Running example: SciPy Integration (Wavelet)")
    print("=========================================================")
    
    # Generate random true parameters
    true_params = generate_random_true_params()
    print("Randomly generated true parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Define the sampling points - use the same for training and prediction
    x_sampling = np.linspace(-10, 10, 200)
    
    # Generate data
    x_data = x_sampling.copy()  # Use the same x points
    y_data = generate_noisy_data(x_data, true_params)
    
    # Use enhanced curve_fit function
    print("Performing curve fitting with automatic parameter estimation...")
    popt, _ = scipy_integration.curve_fit(
        wavelet, x_data, y_data,
        param_ranges={
            'frequency': (0.1, 2.0),
            'phase': (0, 2*np.pi),
            'position': (-5.0, 5.0),
            'width': (0.5, 5.0),
        },
        independent_vars_sampling={
            'x': x_sampling,
        },
    )
    
    # Convert popt to dictionary
    fitted_params = {
        'frequency': popt[0],
        'phase': popt[1],
        'position': popt[2],
        'width': popt[3],
    }
    print("Fitted parameters:")
    for param, value in fitted_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Plot results
    plot_fit_comparison(
        wavelet, x_data, y_data,
        true_params=true_params,
        fitted_params=fitted_params,
        title="Wavelet Fit with SciPy Integration",
    )
    plt.savefig("wavelet_scipy_integration.png")
    
    print("Saved plot to current directory")


def example_lmfit_integration():
    """Example of using ZeroGuess with lmfit integration for a wavelet function."""
    if not LMFIT_AVAILABLE:
        print("\n=========================================================")
        print("Skipping lmfit integration example: lmfit not installed")
        print("=========================================================")
        return

    print("\n=========================================================")
    print("Running example: lmfit Integration (Wavelet)")
    print("=========================================================")
    
    # Generate random true parameters
    true_params = generate_random_true_params()
    print("Randomly generated true parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Define the sampling points - use the same for training and prediction
    x_sampling = np.linspace(-10, 10, 200)
    
    # Generate data
    x_data = x_sampling.copy()  # Use the same x points
    y_data = generate_noisy_data(x_data, true_params)
    
    # Create enhanced lmfit Model with automatic parameter estimation
    print("Creating model with automatic parameter estimation...")
    model = lmfit_integration.Model(
        wavelet,
        param_ranges={
            'frequency': (0.1, 2.0),
            'phase': (0, 2*np.pi),
            'position': (-5.0, 5.0),
            'width': (0.5, 5.0),
        },
        independent_vars_sampling={
            'x': x_sampling,
        },
    )
    
    # Fit data using automatic parameter estimation
    print("Fitting data with automatic parameter estimation...")
    result = model.fit(y_data, x=x_data)
    
    # Extract fitted parameters
    fitted_params = {
        'frequency': result.params['frequency'].value,
        'phase': result.params['phase'].value,
        'position': result.params['position'].value,
        'width': result.params['width'].value,
    }
    
    print("Fitted parameters:")
    for param, value in fitted_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Print fit statistics
    print(f"Fit quality: reduced chi-square = {result.redchi:.6f}")
    
    # Plot results
    plot_fit_comparison(
        wavelet, x_data, y_data,
        true_params=true_params,
        fitted_params=fitted_params,
        title="Wavelet Fit with lmfit Integration",
    )
    plt.savefig("wavelet_lmfit_integration.png")
    
    plot_parameter_comparison(
        true_params, fitted_params,
        title="Wavelet Parameter Comparison (lmfit Integration)",
    )
    plt.savefig("wavelet_parameter_comparison_lmfit.png")
    
    print("Saved plots to current directory")


def example_lmfit_manual():
    """Example of using standard lmfit without automatic parameter estimation."""
    if not LMFIT_AVAILABLE:
        print("\n=========================================================")
        print("Skipping standard lmfit example: lmfit not installed")
        print("=========================================================")
        return

    print("\n=========================================================")
    print("Running example: Standard lmfit Model (Wavelet)")
    print("=========================================================")
    
    # Generate random true parameters
    true_params = generate_random_true_params()
    print("Randomly generated true parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Define the sampling points
    x_sampling = np.linspace(-10, 10, 200)
    
    # Generate data
    x_data = x_sampling.copy()
    y_data = generate_noisy_data(x_data, true_params)
    
    # Create standard lmfit Model
    print("Creating standard lmfit Model...")
    model = lmfit.Model(wavelet)
    
    # Set initial parameter guesses (intentionally off)
    params = model.make_params(
        frequency=0.5,  # Generic initial guess
        phase=0.0,      # Generic initial guess
        position=0.0,   # Generic initial guess
        width=1.0,      # Generic initial guess
    )
    
    # Fit data with standard lmfit
    print("Fitting data with standard lmfit (generic initial guesses)...")
    result = model.fit(y_data, params, x=x_data)
    
    # Extract fitted parameters
    fitted_params = {
        'frequency': result.params['frequency'].value,
        'phase': result.params['phase'].value,
        'position': result.params['position'].value,
        'width': result.params['width'].value,
    }
    
    print("Fitted parameters:")
    for param, value in fitted_params.items():
        print(f"  {param}: {value:.6f}")
    
    # Print fit statistics
    print(f"Fit quality: reduced chi-square = {result.redchi:.6f}")
    
    # Plot results
    plot_fit_comparison(
        wavelet, x_data, y_data,
        true_params=true_params,
        fitted_params=fitted_params,
        title="Wavelet Fit with Standard lmfit",
    )
    plt.savefig("wavelet_standard_lmfit.png")
    
    print("Saved plot to current directory")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Run ZeroGuess example with a Gaussian-enveloped wavelet function'
    )
    parser.add_argument(
        '--random', 
        action='store_true', 
        help='Use truly random parameters (no fixed seed)'
    )
    parser.add_argument(
        '--method', 
        type=str,
        choices=['all', 'basic', 'scipy', 'lmfit'],
        default='all',
        help='Specify which method(s) to run (default: all)'
    )
    args = parser.parse_args()
    
    # Set random seed for reproducibility unless --random flag is used
    if not args.random:
        np.random.seed(123)
        print("Using fixed random seed (123) for reproducible results")
    else:
        print("Using truly random parameters for each run")
    
    # Run selected examples
    if args.method in ['all', 'basic']:
        example_basic_usage()
    
    if args.method in ['all', 'scipy']:
        example_scipy_integration()
    
    if args.method in ['all', 'lmfit']:
        example_lmfit_integration()
        
        # Only run the standard lmfit example if we're specifically running lmfit examples
        if args.method == 'lmfit':
            example_lmfit_manual() 