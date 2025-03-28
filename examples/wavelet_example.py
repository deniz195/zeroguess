#!/usr/bin/env python
"""
Comprehensive example demonstrating ZeroGuess for parameter estimation with a wavelet function.

This example shows three different approaches to using ZeroGuess:
1. Basic usage - Creating and training an estimator directly
2. SciPy integration - Using the enhanced curve_fit function
3. lmfit integration - Using the enhanced Model class

The true parameters are randomly generated once and shared across all methods for better comparison.
The same noisy dataset is used for all methods to ensure direct comparison.

Usage:
    python wavelet_example.py [--random] [--method METHOD]

Options:
    --random         Use truly random parameters (no fixed seed)
    --method METHOD  Specify which method(s) to run: 'all', 'basic', 'scipy', or 'lmfit' (default: 'all')
"""
import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

import zeroguess
from zeroguess.utils.visualization import plot_fit_comparison, plot_parameter_comparison, plot_training_history

# Check if lmfit is installed
try:
    pass

    LMFIT_AVAILABLE = True
except ImportError:
    LMFIT_AVAILABLE = False
    print("Note: lmfit package is not installed. The lmfit example will be skipped.")
    print("To use lmfit integration, install it with: pip install lmfit")

# Add the parent directory to the path so we can import zeroguess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


# Only import ZeroGuessModel if lmfit is available
if LMFIT_AVAILABLE:
    from zeroguess.integration import ZeroGuessModel


def wavelet(x, frequency, phase, position, width):
    """Wavelet function (a modulated Gaussian).

    This function implements a wavelet, which is a wave modulated by a Gaussian envelope.

    Args:
        x: The independent variable
        frequency: The frequency of the wave
        phase: The phase offset of the wave
        position: The center position of the Gaussian envelope
        width: The width of the Gaussian envelope

    Returns:
        The wavelet function values at x
    """
    # Calculate the Gaussian envelope
    envelope = np.exp(-((x - position) ** 2) / (2 * width**2))

    # Calculate the wave
    wave = np.sin(2 * np.pi * frequency * x + phase)

    # Modulate the wave with the envelope
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
    - frequency: 0.5 to 2.0 (reasonable oscillation frequency)
    - phase: 0.0 to 2*pi (full phase range)
    - position: -2.0 to 2.0 (centered within the typical sampling window)
    - width: 1.0 to 3.0 (ensures the envelope is neither too narrow nor too wide)

    Returns:
        Dictionary of randomly generated parameter values
    """
    return {
        "frequency": np.random.uniform(0.5, 2.0),  # oscillation frequency
        "phase": np.random.uniform(0, 2 * np.pi),  # phase offset
        "position": np.random.uniform(-2.0, 2.0),  # center position
        "width": np.random.uniform(1.0, 3.0),  # envelope width
    }


def example_basic_usage(true_params, x_data, y_data, x_sampling=None):
    """Example of basic usage of ZeroGuess.

    Args:
        true_params: Dictionary of true parameter values to use
        x_data: Independent variable values for fitting
        y_data: Dependent variable values (noisy data) for fitting
        x_sampling: Optional pre-defined sampling points for training
    """
    print("\n=========================================================")
    print("Running example: Basic Usage (Wavelet)")
    print("=========================================================")

    print("Using true parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")

    # Create and train estimator
    print("Creating and training estimator...")
    estimator = zeroguess.create_estimator(
        function=wavelet,
        param_ranges={
            "frequency": (0.1, 5.0),
            "phase": (0, 2 * np.pi),
            "position": (-3.0, 3.0),
            "width": (0.1, 5.0),
        },
        independent_vars_sampling={
            "x": x_sampling,
        },
    )

    training_results = estimator.train(
        n_samples=1000,
        n_epochs=3000,
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
    from scipy import optimize

    # Define bounds for curve fitting to help convergence
    bounds = ([0.1, 0, -5, 0.1], [5, 2 * np.pi, 5, 5])

    popt, _ = optimize.curve_fit(
        wavelet,
        x_data,
        y_data,
        p0=[estimated_params[param] for param in ["frequency", "phase", "position", "width"]],
        bounds=bounds,
    )

    # Convert popt to dictionary
    fitted_params = {
        "frequency": popt[0],
        "phase": popt[1],
        "position": popt[2],
        "width": popt[3],
    }
    print("Fitted parameters:")
    for param, value in fitted_params.items():
        print(f"  {param}: {value:.6f}")

    # Plot results
    plot_training_history(
        training_results["train_losses"],
        training_results["val_losses"],
        title="Training History: Wavelet Parameter Estimation",
    )
    plt.savefig("wavelet_training_history.png")

    plot_fit_comparison(
        wavelet,
        x_data,
        y_data,
        true_params=true_params,
        estimated_params=estimated_params,
        fitted_params=fitted_params,
        title="Wavelet Fit Comparison (Basic Usage)",
    )
    plt.savefig("wavelet_fit_comparison.png")

    plot_parameter_comparison(
        true_params,
        estimated_params,
        fitted_params,
        title="Wavelet Parameter Comparison (Basic Usage)",
    )
    plt.savefig("wavelet_parameter_comparison.png")

    print("Saved plots to current directory")
    return x_sampling  # Return sampling points for reuse


def example_lmfit_integration(true_params, x_data, y_data, x_sampling=None):
    """Example of using ZeroGuess with lmfit integration.

    Args:
        true_params: Dictionary of true parameter values to use
        x_data: Independent variable values for fitting
        y_data: Dependent variable values (noisy data) for fitting
        x_sampling: Optional pre-defined sampling points for training
    """
    if not LMFIT_AVAILABLE:
        print("\n=========================================================")
        print("Skipping lmfit integration example: lmfit not installed")
        print("=========================================================")
        return

    print("\n=========================================================")
    print("Running example: lmfit Integration (Wavelet)")
    print("=========================================================")

    print("Using true parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")

    # Create enhanced lmfit Model with automatic parameter estimation
    print("Creating model with ZeroGuess integration...")
    model = ZeroGuessModel(
        wavelet,
        independent_vars_sampling={
            "x": x_sampling,
        },
    )

    # Set parameter bounds to enable zeroguess
    model.set_param_hint("frequency", min=0.1, max=5.0)
    model.set_param_hint("phase", min=0, max=2 * np.pi)
    model.set_param_hint("position", min=-5, max=5)
    model.set_param_hint("width", min=0.5, max=5)  # Set a more strict lower bound for width
    model.make_params()

    # Using guess() method directly to demonstrate how it works
    print("Demonstrating guess() method directly...")

    # Get parameters directly from guess()
    guessed_params = model.guess(y_data, x=x_data)

    print("Parameters from guess() method:")
    for param, value in guessed_params.items():
        true_value = true_params[param]
        print(f"  {param}: {value.value:.6f} (true: {true_value:.6f})")

    # Fit using no initial parameters - will use guess() automatically
    print("Fitting without explicit parameters (using guess() automatically)...")
    result = model.fit(y_data, params=guessed_params, x=x_data)  # No params provided

    # Extract fitted parameters
    fitted_params = result.params

    print("Fitted parameters:")
    for param, value in fitted_params.items():
        true_value = true_params[param]
        print(f"  {param}: {value.value:.6f} (true: {true_value:.6f})")

    # Print fit statistics
    print(f"Fit quality: reduced chi-square = {result.redchi:.6f}")

    # Plot results
    plot_fit_comparison(
        wavelet,
        x_data,
        y_data,
        true_params=true_params,
        estimated_params=guessed_params,
        fitted_params=fitted_params,
        title="Wavelet Fit with lmfit Integration",
    )
    plt.savefig("wavelet_fit_lmfit.png")

    # Plot parameter comparison
    plot_parameter_comparison(
        true_params,
        guessed_params,
        fitted_params,
        title="Wavelet Parameter Comparison (lmfit Integration)",
    )
    plt.savefig("wavelet_parameter_comparison_lmfit.png")

    print("Saved plots to current directory")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run comprehensive ZeroGuess examples with a wavelet function")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use truly random parameters (no fixed seed)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["all", "basic", "lmfit"],
        default="all",
        help="Specify which method(s) to run (default: all)",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility unless --random flag is used
    if not args.random:
        np.random.seed(123)
        print("Using fixed random seed (123) for reproducible results")
    else:
        print("Using truly random parameters for each run")

    # Generate true parameters once to use across all examples
    true_params = generate_random_true_params()
    print("\nGenerated true parameters that will be used for all methods:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")

    # Define common sampling points for both training and fitting
    x_sampling = np.linspace(-10, 10, 300)

    # Generate noisy data once to use across all examples
    x_data = x_sampling.copy()
    y_data = generate_noisy_data(x_data, true_params)
    print("\nGenerated noisy data that will be used for all methods")

    # Run selected examples
    if args.method in ["all", "basic"]:
        x_sampling = example_basic_usage(true_params, x_data, y_data, x_sampling)

    if args.method in ["all", "lmfit"]:
        # Will handle running example_lmfit_manual internally if needed
        example_lmfit_integration(true_params, x_data, y_data, x_sampling)
