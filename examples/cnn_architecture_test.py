#!/usr/bin/env python
"""
CNN Architecture Test Script for ZeroGuess

This script demonstrates and tests the CNN architecture implementation
in ZeroGuess for parameter estimation of oscillatory functions.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time

from zeroguess import create_estimator


def damped_sine(x, amplitude, frequency, phase, decay):
    """
    Damped sine wave function, which is particularly suitable for CNN architecture.
    
    Args:
        x: Independent variable
        amplitude: Peak amplitude
        frequency: Oscillation frequency
        phase: Phase offset
        decay: Exponential decay rate
        
    Returns:
        Damped sine wave values at x positions
    """
    return amplitude * np.sin(2 * np.pi * frequency * x + phase) * np.exp(-decay * x)


def main(args):
    """
    Main function to test CNN architecture.
    
    Args:
        args: Command line arguments
    """
    print(f"Testing CNN architecture with {args.architecture}")

    # Define independent variable for training and prediction
    # Using a consistent number of data points (200) for both training and prediction
    x_sampling = np.linspace(0, 10, 200)
    
    # Create parameter estimator
    estimator = create_estimator(
        function=damped_sine,
        param_ranges={
            'amplitude': (0.5, 5.0),
            'frequency': (0.5, 3.0),
            'phase': (0, 2 * np.pi),
            'decay': (0.1, 1.0)
        },
        independent_vars_sampling={
            'x': x_sampling
        },
        estimator_type="neural_network",
        architecture=args.architecture,
        architecture_params=args.architecture_params,
    )
    
    # Get and display available architectures
    available_architectures = estimator.list_available_architectures()
    print(f"Available architectures: {available_architectures}")
    
    # Train the estimator
    if args.train:
        print("\nTraining the estimator...")
        start_time = time.time()
        history = estimator.train(
            n_samples=args.samples,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            validation_split=0.2,
            verbose=True
        )
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training History ({args.architecture.upper()} Architecture)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'cnn_training_history_{args.architecture}.png')
        print(f"Training history saved to cnn_training_history_{args.architecture}.png")
        
        # Save the model
        model_file = f'damped_sine_estimator_{args.architecture}.pt'
        estimator.save(model_file)
        print(f"Model saved to {model_file}")
    
    # Use the estimator for prediction
    if args.predict or not args.train:
        if not args.train:
            # If not training, try to load a pre-trained model
            try:
                model_file = f'damped_sine_estimator_{args.architecture}.pt'
                print(f"\nLoading pre-trained model from {model_file}...")
                estimator = estimator.__class__.load(model_file)
            except FileNotFoundError:
                print(f"No pre-trained model found. Training a new model...")
                history = estimator.train(
                    n_samples=args.samples,
                    batch_size=args.batch_size,
                    n_epochs=args.epochs,
                    validation_split=0.2,
                    verbose=True
                )
        
        # Generate test data with known parameters
        print("\nGenerating test data with known parameters...")
        real_params = {
            'amplitude': 2.5,
            'frequency': 1.2,
            'phase': 0.8,
            'decay': 0.3
        }
        
        # Use the SAME x_sampling as used in training to avoid input size mismatch
        y_data = damped_sine(x_sampling, **real_params)
        y_data_noisy = y_data + np.random.normal(0, 0.05, len(y_data))
        
        # Predict parameters
        print("Predicting parameters...")
        start_time = time.time()
        predicted_params = estimator.predict(x=x_sampling, y=y_data_noisy)
        prediction_time = time.time() - start_time
        print(f"Prediction completed in {prediction_time:.4f} seconds")
        
        # Print results
        print("\nReal parameters:")
        for param, value in real_params.items():
            print(f"  - {param}: {value}")
        
        print("\nPredicted parameters:")
        for param, value in predicted_params.items():
            print(f"  - {param}: {value:.4f}")
            
        # Calculate parameter errors
        errors = {}
        for param in real_params:
            errors[param] = abs(real_params[param] - predicted_params[param]) / real_params[param] * 100
        
        print("\nParameter errors (%):")
        for param, error in errors.items():
            print(f"  - {param}: {error:.2f}%")
            
        # Plot the results
        plt.figure(figsize=(12, 6))
        plt.plot(x_sampling, y_data, 'k-', label='True function')
        plt.plot(x_sampling, y_data_noisy, 'b.', alpha=0.3, label='Noisy data')
        plt.plot(x_sampling, damped_sine(x_sampling, **predicted_params), 'r-', label='Predicted parameters')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title(f'Parameter Estimation ({args.architecture.upper()} Architecture)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'cnn_parameter_estimation_{args.architecture}.png')
        print(f"Plot saved to cnn_parameter_estimation_{args.architecture}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Architecture Test for ZeroGuess")
    
    parser.add_argument(
        "--architecture", 
        type=str, 
        default="cnn",
        choices=["mlp", "cnn", "best"],
        help="Neural network architecture to use"
    )
    
    parser.add_argument(
        "--train", 
        action="store_true",
        help="Train a new model (otherwise tries to load a pre-trained model)"
    )
    
    parser.add_argument(
        "--predict", 
        action="store_true",
        help="Perform prediction using the model"
    )
    
    parser.add_argument(
        "--samples", 
        type=int, 
        default=1000,
        help="Number of training samples to generate"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=64,
        help="Batch size for training"
    )
    
    # Convert architecture params from command line to a dictionary
    parser.add_argument(
        "--architecture-params",
        type=lambda params: {
            param.split(':')[0]: eval(param.split(':')[1])
            for param in params.split() if ':' in param
        },
        default={},
        help="Architecture-specific parameters (format: 'param:value param2:value2')"
    )
    
    args = parser.parse_args()
    main(args) 