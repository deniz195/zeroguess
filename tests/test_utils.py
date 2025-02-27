"""Utility functions for ZeroGuess tests."""

import numpy as np


def calculate_parameter_error(estimated_params, true_params):
    """Calculate relative errors between estimated and true parameters.

    Args:
        estimated_params (dict): Dictionary of estimated parameters
        true_params (dict): Dictionary of true parameters

    Returns:
        dict: Dictionary of relative errors for each parameter
    """
    errors = {}
    for param_name, true_value in true_params.items():
        if param_name not in estimated_params:
            errors[param_name] = float("inf")
            continue

        estimated_value = estimated_params[param_name]

        # Handle zero or near-zero true values
        if abs(true_value) < 1e-10:
            # Use absolute error instead of relative error
            errors[param_name] = abs(estimated_value - true_value)
        else:
            # Calculate relative error
            errors[param_name] = abs((estimated_value - true_value) / true_value)

    return errors


def calculate_curve_fit_quality(func, x_data, y_data, params):
    """Calculate the quality of curve fit using the given parameters.

    Args:
        func (callable): The function to fit
        x_data (array): Independent variable data
        y_data (array): Dependent variable data
        params (dict): Parameters to use in the function

    Returns:
        float: Root mean square error (RMSE) of the fit
    """
    # Generate y values using the function and parameters
    y_fit = func(x_data, **params)

    # Calculate RMSE
    residuals = y_data - y_fit
    rmse = np.sqrt(np.mean(residuals**2))

    return rmse


def is_within_tolerance(errors, tolerance=0.1):
    """Check if all parameter errors are within the specified tolerance.

    Args:
        errors (dict): Dictionary of parameter errors
        tolerance (float): Maximum allowed relative error

    Returns:
        bool: True if all errors are within tolerance, False otherwise
    """
    return all(error < tolerance for error in errors.values())


def generate_noisy_data(func, x_data, params, noise_level=0.1, random_seed=None):
    """Generate noisy data from a function with known parameters.

    Args:
        func (callable): The function to generate data from
        x_data (array): Independent variable data points
        params (dict): Parameters to use in the function
        noise_level (float): Standard deviation of Gaussian noise to add
        random_seed (int, optional): Seed for reproducible noise generation

    Returns:
        array: Noisy dependent variable data
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate clean data
    y_clean = func(x_data, **params)

    # Add Gaussian noise
    noise = np.random.normal(0, noise_level, size=len(x_data))
    y_noisy = y_clean + noise

    return y_noisy


def calculate_snr(signal, noise):
    """Calculate signal-to-noise ratio.

    Args:
        signal (array): Clean signal values
        noise (array): Noise values

    Returns:
        float: Signal-to-noise ratio
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)

    if noise_power < 1e-10:  # Avoid division by zero
        return float("inf")

    return signal_power / noise_power
