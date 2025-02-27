#!/usr/bin/env python
"""
Comprehensive example demonstrating ZeroGuess for parameter estimation with a Gaussian function.

This example shows three different approaches to using ZeroGuess:
1. Basic usage - Creating and training an estimator directly
2. SciPy integration - Using the enhanced curve_fit function
3. lmfit integration - Using the enhanced Model class

The true parameters are randomly generated once and shared across all methods for better comparison.
The same noisy dataset is used for all methods to ensure direct comparison.

Usage:
    python comprehensive_example.py [--random] [--method METHOD]

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
from zeroguess.integration import scipy_integration
from zeroguess.utils.visualization import (
    plot_fit_comparison,
    plot_parameter_comparison,
    plot_training_history,
)

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


# Only import lmfit_integration if lmfit is available
if LMFIT_AVAILABLE:
    from zeroguess.integration import lmfit_integration


def gaussian(x, amplitude, center, width):
    """Gaussian function.

    Args:
        x: The independent variable
        amplitude: The peak height of the Gaussian
        center: The center position of the Gaussian
        width: The standard deviation of the Gaussian

    Returns:
        The Gaussian function values at x
    """
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width**2))


def generate_noisy_data(x, true_params, noise_level=0.1):
    """Generate noisy data for testing.

    Args:
        x: Independent variable values
        true_params: Dictionary of true parameter values
        noise_level: Relative noise level as a fraction of the signal amplitude

    Returns:
        Noisy data points
    """
    y_true = gaussian(x, **true_params)
    noise = np.random.normal(0, noise_level * np.max(y_true), size=len(x))
    return y_true + noise


def generate_random_true_params():
    """Generate random true parameters within reasonable ranges.

    The parameters are randomly chosen within these ranges:
    - amplitude: 1.0 to 8.0 (reasonable peak height)
    - center: -3.0 to 3.0 (centered within the typical sampling window)
    - width: 0.5 to 3.0 (ensures the Gaussian is neither too narrow nor too wide)

    Returns:
        Dictionary of randomly generated parameter values
    """
    return {
        "amplitude": np.random.uniform(1.0, 8.0),  # peak height
        "center": np.random.uniform(-3.0, 3.0),  # center position
        "width": np.random.uniform(0.5, 3.0),  # peak width
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
    print("Running example: Basic Usage (Gaussian)")
    print("=========================================================")

    print("Using true parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")

    # Create and train estimator
    print("Creating and training estimator...")
    estimator = zeroguess.create_estimator(
        function=gaussian,
        param_ranges={
            "amplitude": (0.1, 10.0),
            "center": (-5.0, 5.0),
            "width": (0.1, 5.0),
        },
        independent_vars_sampling={
            "x": x_sampling,
        },
    )

    training_results = estimator.train(
        n_samples=1000,
        epochs=1000,
        batch_size=32,
        add_noise=True,
        noise_level=0.1,
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

    popt, _ = optimize.curve_fit(
        gaussian,
        x_data,
        y_data,
        p0=[estimated_params[param] for param in ["amplitude", "center", "width"]],
    )

    # Convert popt to dictionary
    fitted_params = {
        "amplitude": popt[0],
        "center": popt[1],
        "width": popt[2],
    }
    print("Fitted parameters:")
    for param, value in fitted_params.items():
        print(f"  {param}: {value:.6f}")

    # Plot results
    plot_training_history(
        training_results["train_losses"],
        training_results["val_losses"],
        title="Training History: Gaussian Parameter Estimation",
    )
    plt.savefig("gaussian_training_history.png")

    plot_fit_comparison(
        gaussian,
        x_data,
        y_data,
        true_params=true_params,
        estimated_params=estimated_params,
        fitted_params=fitted_params,
        title="Gaussian Fit Comparison (Basic Usage)",
    )
    plt.savefig("gaussian_fit_comparison_basic.png")

    plot_parameter_comparison(
        true_params,
        estimated_params,
        fitted_params,
        title="Gaussian Parameter Comparison (Basic Usage)",
    )
    plt.savefig("gaussian_parameter_comparison_basic.png")

    print("Saved plots to current directory")
    return x_sampling  # Return sampling points for reuse


def example_scipy_integration(true_params, x_data, y_data, x_sampling=None):
    """Example of using ZeroGuess with SciPy integration.

    Args:
        true_params: Dictionary of true parameter values to use
        x_data: Independent variable values for fitting
        y_data: Dependent variable values (noisy data) for fitting
        x_sampling: Optional pre-defined sampling points for training
    """
    print("\n=========================================================")
    print("Running example: SciPy Integration (Gaussian)")
    print("=========================================================")

    print("Using true parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")

    # Use enhanced curve_fit function
    print("Performing curve fitting with automatic parameter estimation...")
    popt, _ = scipy_integration.curve_fit(
        gaussian,
        x_data,
        y_data,
        param_ranges={
            "amplitude": (0.1, 10.0),
            "center": (-5.0, 5.0),
            "width": (0.1, 5.0),
        },
        independent_vars_sampling={
            "x": x_sampling,
        },
    )

    # Convert popt to dictionary
    fitted_params = {
        "amplitude": popt[0],
        "center": popt[1],
        "width": popt[2],
    }
    print("Fitted parameters:")
    for param, value in fitted_params.items():
        print(f"  {param}: {value:.6f}")

    # Plot results
    plot_fit_comparison(
        gaussian,
        x_data,
        y_data,
        true_params=true_params,
        fitted_params=fitted_params,
        title="Gaussian Fit with SciPy Integration",
    )
    plt.savefig("gaussian_scipy_integration.png")

    plot_parameter_comparison(
        true_params,
        fitted_params,
        title="Gaussian Parameter Comparison (SciPy Integration)",
    )
    plt.savefig("gaussian_parameter_comparison_scipy.png")

    print("Saved plots to current directory")


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
    print("Running example: lmfit Integration (Gaussian)")
    print("=========================================================")

    print("Using true parameters:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")

    # Create enhanced lmfit Model with automatic parameter estimation
    print("Creating model with automatic parameter estimation...")
    model = lmfit_integration.Model(
        gaussian,
        # param_ranges={
        #     'amplitude': (0.1, 10.0),
        #     'center': (-5.0, 5.0),
        #     'width': (0.1, 5.0),
        # },
        independent_vars_sampling={
            "x": x_sampling,
        },
        auto_extract_bounds=True,  # Enable automatic extraction of bounds from params
    )

    # Set parameter bounds to help convergence
    # These bounds will also be used for parameter estimation due to auto_extract_bounds=True
    model.set_param_hint("amplitude", min=0.1, max=10.0)
    model.set_param_hint("center", min=-5.0, max=5.0)
    model.set_param_hint("width", min=0.1, max=5.0)
    model.make_params()

    # Using guess() method directly to demonstrate how it works
    print("Demonstrating guess() method directly...")

    # Get parameters directly from guess()
    guessed_params = model.guess(y_data, x=x_data)

    print("Parameters from guess() method:")
    for param, value in guessed_params.items():
        true_value = true_params[param]
        print(f"  {param}: {value.value:.6f} (true: {true_value:.6f})")

    # Fit data using automatic parameter estimation
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
        gaussian,
        x_data,
        y_data,
        true_params=true_params,
        estimated_params=guessed_params,
        fitted_params=fitted_params,
        title="Gaussian Fit with lmfit Integration",
    )
    plt.savefig("gaussian_fit_lmfit.png")

    plot_parameter_comparison(
        true_params,
        guessed_params,
        fitted_params,
        title="Gaussian Parameter Comparison (lmfit Integration)",
    )
    plt.savefig("gaussian_parameter_comparison_lmfit.png")

    print("Saved plots to current directory")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run comprehensive ZeroGuess examples with a Gaussian function")
    parser.add_argument(
        "--random",
        action="store_true",
        help="Use truly random parameters (no fixed seed)",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["all", "basic", "scipy", "lmfit"],
        default="all",
        help="Specify which method(s) to run (default: all)",
    )
    args = parser.parse_args()

    # Set random seed for reproducibility unless --random flag is used
    if not args.random:
        np.random.seed(42)
        print("Using fixed random seed (42) for reproducible results")
    else:
        print("Using truly random parameters for each run")

    # Generate true parameters once to use across all examples
    true_params = generate_random_true_params()
    print("\nGenerated true parameters that will be used for all methods:")
    for param, value in true_params.items():
        print(f"  {param}: {value:.6f}")

    # Define common sampling points for both training and fitting
    x_sampling = np.linspace(-10, 10, 100)

    # Generate noisy data once to use across all examples
    x_data = x_sampling.copy()
    y_data = generate_noisy_data(x_data, true_params)
    print("\nGenerated noisy data that will be used for all methods")

    # Run selected examples
    if args.method in ["all", "basic"]:
        x_sampling = example_basic_usage(true_params, x_data, y_data, x_sampling)

    if args.method in ["all", "scipy"]:
        example_scipy_integration(true_params, x_data, y_data, x_sampling)

    if args.method in ["all", "lmfit"]:
        example_lmfit_integration(true_params, x_data, y_data, x_sampling)
