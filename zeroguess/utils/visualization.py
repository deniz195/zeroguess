"""
Visualization utilities for ZeroGuess.
"""

from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_fit_comparison(
    function: Callable,
    x_data: np.ndarray,
    y_data: np.ndarray,
    true_params: Optional[Dict[str, float]] = None,
    estimated_params: Dict[str, float] = None,
    fitted_params: Optional[Dict[str, float]] = None,
    title: str = "Fit Comparison",
    figsize: Tuple[int, int] = (12, 8),
) -> plt.Figure:
    """Plot a comparison of the data, true function, initial estimated fit, and final fit.

    Args:
        function: The curve fitting function
        x_data: Independent variable data
        y_data: Dependent variable data (measured)
        true_params: Dictionary of true parameter values (optional)
        estimated_params: Dictionary of estimated initial parameter values
        fitted_params: Dictionary of final fitted parameter values (optional)
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data points
    ax.scatter(x_data, y_data, color="black", label="Data", alpha=0.5)

    # Generate smooth x values for curves
    x_smooth = np.linspace(np.min(x_data), np.max(x_data), 500)

    # Plot true curve if true parameters are provided
    if true_params is not None:
        y_true = function(x_smooth, **true_params)
        ax.plot(x_smooth, y_true, "g-", linewidth=2, label="True")

    # Plot estimated curve
    if estimated_params is not None:
        y_estimated = function(x_smooth, **estimated_params)
        ax.plot(x_smooth, y_estimated, "r--", linewidth=2, label="Initial Estimate")

    # Plot fitted curve
    if fitted_params is not None:
        y_fitted = function(x_smooth, **fitted_params)
        ax.plot(x_smooth, y_fitted, "b-", linewidth=2, label="Final Fit")

    # Add legend, grid, and title
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_training_history(
    train_losses: List[float],
    val_losses: Optional[List[float]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Training History",
) -> plt.Figure:
    """Plot the training history of a model.

    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch (optional)
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)

    # Plot training loss
    ax.plot(epochs, train_losses, "b-", linewidth=2, label="Training Loss")

    # Plot validation loss if provided
    if val_losses is not None:
        ax.plot(epochs, val_losses, "r--", linewidth=2, label="Validation Loss")

    # Add legend, grid, and labels
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")

    # Set y-axis to logarithmic if the losses span multiple orders of magnitude
    if max(train_losses) / (min(train_losses) + 1e-10) > 100:
        ax.set_yscale("log")

    plt.tight_layout()
    return fig


def plot_parameter_comparison(
    true_params: Dict[str, float],
    estimated_params: Dict[str, float],
    fitted_params: Optional[Dict[str, float]] = None,
    figsize: Tuple[int, int] = (12, 6),
    title: str = "Parameter Comparison",
) -> plt.Figure:
    """Plot a comparison of true, estimated, and fitted parameter values.

    Args:
        true_params: Dictionary of true parameter values
        estimated_params: Dictionary of estimated initial parameter values
        fitted_params: Dictionary of final fitted parameter values (optional)
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    # Get parameter names and ensure they match
    param_names = list(true_params.keys())
    for params in [estimated_params, fitted_params]:
        if params is not None:
            if set(params.keys()) != set(param_names):
                raise ValueError("Parameter dictionaries must have the same keys")

    # Number of parameters
    n_params = len(param_names)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Set up x-coordinates for the bars
    x = np.arange(n_params)
    width = 0.25  # Width of the bars

    # Plot bars for each parameter set
    true_values = [true_params[name] for name in param_names]
    rects1 = ax.bar(x - width, true_values, width, label="True")

    estimated_values = [estimated_params[name] for name in param_names]
    rects2 = ax.bar(x, estimated_values, width, label="Estimated")

    if fitted_params is not None:
        fitted_values = [fitted_params[name] for name in param_names]
        rects3 = ax.bar(x + width, fitted_values, width, label="Fitted")

    # Add labels, title, and legend
    ax.set_xlabel("Parameter")
    ax.set_ylabel("Value")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.legend()

    # Add value labels on the bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.3f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    autolabel(rects1)
    autolabel(rects2)
    if fitted_params is not None:
        autolabel(rects3)

    plt.tight_layout()
    return fig
