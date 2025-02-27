#!/usr/bin/env python
"""
Architecture Selection Example for ZeroGuess

This example demonstrates how to use different neural network architectures
for parameter estimation in ZeroGuess.

Currently supported architectures:
- "mlp" (default): Multilayer Perceptron
- "cnn" (future work): Convolutional Neural Network
- "transformer" (future work): Transformer with self-attention

Note: Only the MLP architecture is fully implemented at this time.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from zeroguess import create_estimator


def gaussian(x, amplitude, center, width):
    """
    Gaussian function for curve fitting.

    Args:
        x: Independent variable
        amplitude: Peak height
        center: Peak center
        width: Peak width (standard deviation)

    Returns:
        Gaussian function values at x positions
    """
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width**2))


def main(args):  # noqa: C901
    """
    Main function to demonstrate architecture selection.

    Args:
        args: Command line arguments
    """
    print(f"Using architecture: {args.architecture}")

    # Define independent variable for training
    x_sampling = np.linspace(-10, 10, 200)

    # Create parameter estimator
    estimator = create_estimator(
        function=gaussian,
        param_ranges={"amplitude": (0.5, 10), "center": (-5, 5), "width": (0.5, 3)},
        independent_vars_sampling={"x": x_sampling},
        estimator_type="neural_network",
        architecture=args.architecture,
        architecture_params=args.architecture_params,
        learning_rate=0.001,
        weight_decay=0.0001,
    )

    # Get and display available architectures
    available_architectures = estimator.list_available_architectures()
    print(f"Available architectures: {available_architectures}")

    # Get architecture details
    architecture_details = estimator.get_architecture_details()
    print("\nArchitecture details:")
    for name, details in architecture_details.items():
        print(f"  - {name}:")
        print(f"    Description: {details['description']}")
        print(f"    Default parameters:")  # noqa: F541
        for param, value in details["default_params"].items():
            print(f"      - {param}: {value}")

    # Train the estimator
    if args.train:
        print("\nTraining the estimator...")
        history = estimator.train(
            n_samples=args.samples,
            batch_size=args.batch_size,
            n_epochs=args.epochs,
            validation_split=0.2,
            verbose=True,
        )

        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history["train_loss"], label="Training Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training History ({args.architecture.upper()} Architecture)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"training_history_{args.architecture}.png")
        print(f"Training history saved to training_history_{args.architecture}.png")

        # Save the model
        model_file = f"gaussian_estimator_{args.architecture}.pt"
        estimator.save(model_file)
        print(f"Model saved to {model_file}")

    # Use the estimator for prediction
    if args.predict or not args.train:
        if not args.train:
            # If not training, try to load a pre-trained model
            try:
                model_file = f"gaussian_estimator_{args.architecture}.pt"
                print(f"\nLoading pre-trained model from {model_file}...")
                estimator = estimator.__class__.load(model_file)
            except FileNotFoundError:
                print("No pre-trained model found. Training a new model...")
                history = estimator.train(
                    n_samples=args.samples,
                    batch_size=args.batch_size,
                    n_epochs=args.epochs,
                    validation_split=0.2,
                    verbose=True,
                )

        # Generate test data with known parameters
        print("\nGenerating test data with known parameters...")
        real_params = {"amplitude": 5.0, "center": 1.5, "width": 1.2}

        # Create x data with some noise
        x_data = np.linspace(-10, 10, 200)
        y_data = gaussian(x_data, **real_params)
        y_data_noisy = y_data + np.random.normal(0, 0.1, len(y_data))

        # Predict parameters
        print("Predicting parameters...")
        predicted_params = estimator.predict(x=x_data, y=y_data_noisy)

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
        plt.plot(x_data, y_data, "k-", label="True function")
        plt.plot(x_data, y_data_noisy, "b.", alpha=0.3, label="Noisy data")
        plt.plot(
            x_data,
            gaussian(x_data, **predicted_params),
            "r-",
            label="Predicted parameters",
        )
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(f"Parameter Estimation ({args.architecture.upper()} Architecture)")
        plt.legend()
        plt.grid(True)
        plt.savefig(f"parameter_estimation_{args.architecture}.png")
        print(f"Plot saved to parameter_estimation_{args.architecture}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of architecture selection in ZeroGuess")

    parser.add_argument(
        "--architecture",
        type=str,
        default="mlp",
        choices=["mlp", "cnn", "transformer", "best"],
        help="Neural network architecture to use",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train a new model (otherwise tries to load a pre-trained model)",
    )

    parser.add_argument("--predict", action="store_true", help="Perform prediction using the model")

    parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of training samples to generate",
    )

    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")

    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")

    # Convert architecture params from command line to a dictionary
    # Format: --architecture-params hidden_layers:[128,256,128] activation:relu dropout_rate:0.1
    parser.add_argument(
        "--architecture-params",
        type=lambda params: {
            param.split(":")[0]: eval(param.split(":")[1]) for param in params.split() if ":" in param
        },
        default={},
        help="Architecture-specific parameters (format: 'param:value param2:value2')",
    )

    args = parser.parse_args()
    main(args)
