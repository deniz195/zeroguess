"""
Utility functions for working with fitting functions.

This module provides utility functions for working with fitting functions,
such as adding noise to generated data.
"""

from typing import Optional

import numpy as np


def add_gaussian_noise(
    data: np.ndarray,
    sigma: float = 0.1,
    relative: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Add Gaussian noise to data.

    Args:
        data: The data to add noise to.
        sigma: The standard deviation of the noise. If relative is True,
            this is interpreted as a fraction of the data range.
        relative: If True, sigma is interpreted as a fraction of the data range.
            If False, sigma is interpreted as an absolute value.
        seed: Random seed for reproducibility. If None, a random seed is used.

    Returns:
        The data with added noise.
    """
    # Set random seed if provided
    if seed is not None:
        np.random.seed(seed)

    # Calculate noise level
    if relative:
        # Scale sigma by the data range
        data_range = np.max(data) - np.min(data)
        if data_range == 0:
            # Avoid division by zero for constant data
            noise_level = sigma
        else:
            noise_level = sigma * data_range
    else:
        # Use absolute sigma value
        noise_level = sigma

    # Generate and add noise
    noise = np.random.normal(0, noise_level, size=data.shape)
    return data + noise


def signal_to_noise_ratio(original_data: np.ndarray, noisy_data: np.ndarray) -> float:
    """Calculate the signal-to-noise ratio.

    Args:
        original_data: The original clean data.
        noisy_data: The data with added noise.

    Returns:
        The signal-to-noise ratio in decibels.
    """
    # Calculate signal power
    signal_power = np.mean(original_data**2)

    # Calculate noise power
    noise = noisy_data - original_data
    noise_power = np.mean(noise**2)

    # Avoid division by zero
    if noise_power == 0:
        return float("inf")

    # Calculate SNR in decibels
    snr = 10 * np.log10(signal_power / noise_power)
    return snr
