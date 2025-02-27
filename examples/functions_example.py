#!/usr/bin/env python
"""
Example demonstrating the use of the ZeroGuess functions submodule.

This script shows how to use the pre-defined fitting functions and utilities
provided by the zeroguess.functions submodule.
"""

import numpy as np
import matplotlib.pyplot as plt
from zeroguess.functions import (
    GaussianFunction,
    MultiPeakGaussianFunction,
    DampedSineFunction,
    LinearFunction,
    add_gaussian_noise
)


def main():
    """Run the example code."""
    # Create instances of the different function types
    gaussian = GaussianFunction()
    multi_peak = MultiPeakGaussianFunction(n_peaks=2)
    damped_sine = DampedSineFunction()
    linear = LinearFunction()
    
    # Print information about each function
    print("=== Available Functions ===")
    print(f"1. {gaussian.name}")
    print(f"2. {multi_peak.name}")
    print(f"3. {damped_sine.name}")
    print(f"4. {linear.name}")
    print()
    
    # Example 1: Working with Gaussian function
    print("=== Gaussian Function Example ===")
    print("Parameter ranges:")
    for name, (min_val, max_val) in gaussian.param_ranges.items():
        print(f"  - {name}: [{min_val}, {max_val}]")
    
    print("\nParameter descriptions:")
    for name, desc in gaussian.param_descriptions.items():
        print(f"  - {name}: {desc}")
    
    # Generate data with default parameters
    params = {
        'amplitude': 3.0,
        'center': 0.0,
        'width': 1.0
    }
    indep_vars, y_data = gaussian.generate_data(params)
    x = indep_vars['x']
    
    # Add noise to the data
    y_noisy = add_gaussian_noise(y_data, sigma=0.05)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_data, 'b-', label='Clean Data')
    plt.plot(x, y_noisy, 'r.', alpha=0.5, label='Noisy Data')
    plt.title('Gaussian Function Example')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('gaussian_example.png')
    print("\nPlot saved to gaussian_example.png")
    
    # Example 2: Working with Multi-peak Gaussian function
    print("\n=== Multi-peak Gaussian Function Example ===")
    
    # Generate data with custom parameters
    params = {
        'amplitude_1': 5.0,
        'center_1': -2.0,
        'width_1': 0.8,
        'amplitude_2': 3.0,
        'center_2': 2.0,
        'width_2': 1.2
    }
    indep_vars, y_data = multi_peak.generate_data(params)
    x = indep_vars['x']
    
    # Add noise to the data
    y_noisy = add_gaussian_noise(y_data, sigma=0.03)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_data, 'b-', label='Clean Data')
    plt.plot(x, y_noisy, 'r.', alpha=0.5, label='Noisy Data')
    plt.title('Multi-peak Gaussian Function Example')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('multi_peak_example.png')
    print("Plot saved to multi_peak_example.png")
    
    # Example 3: Working with Damped Sine function
    print("\n=== Damped Sine Function Example ===")
    
    # Use random parameters
    random_params = damped_sine.get_random_params()
    print("Random parameters:")
    for name, value in random_params.items():
        print(f"  - {name}: {value:.4f}")
    
    indep_vars, y_data = damped_sine.generate_data(random_params)
    x = indep_vars['x']
    
    # Add noise to the data
    y_noisy = add_gaussian_noise(y_data, sigma=0.05)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_data, 'b-', label='Clean Data')
    plt.plot(x, y_noisy, 'r.', alpha=0.5, label='Noisy Data')
    plt.title('Damped Sine Function Example')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('damped_sine_example.png')
    print("Plot saved to damped_sine_example.png")
    
    # Example 4: Working with Linear function
    print("\n=== Linear Function Example ===")
    
    # Get default independent variable sampling
    indep_vars = linear.default_independent_vars
    x = indep_vars['x']
    
    # Define parameters
    params = {
        'slope': 0.5,
        'intercept': -1.0
    }
    
    # Evaluate the function
    y_data = linear(x, **params)
    
    # Add noise to the data
    y_noisy = add_gaussian_noise(y_data, sigma=0.1)
    
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x, y_data, 'b-', label='Clean Data')
    plt.plot(x, y_noisy, 'r.', alpha=0.5, label='Noisy Data')
    plt.title('Linear Function Example')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.savefig('linear_example.png')
    print("Plot saved to linear_example.png")
    
    print("\nAll examples completed.")


if __name__ == "__main__":
    main() 