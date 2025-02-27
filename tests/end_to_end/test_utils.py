"""Utility functions for end-to-end tests."""

import datetime
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from zeroguess.utils.visualization import (
    plot_fit_comparison,
    plot_parameter_comparison,
    plot_training_history,
)

# Define the output directory for visualization files
VISUALIZATION_OUTPUT_DIR = os.path.join("tests", "output", "visualizations")


def get_timestamp():
    """Generate a timestamp string for unique filenames."""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_output_dir_exists():
    """Create the visualization output directory if it doesn't exist."""
    pathlib.Path(VISUALIZATION_OUTPUT_DIR).mkdir(parents=True, exist_ok=True)


def create_and_save_visualizations(
    function, x_data, y_data, true_params, estimated_params, test_name, monkeypatch=None
):
    """Create visualization figures and save output files.

    Args:
        function (callable): The function used for curve fitting.
        x_data (numpy.ndarray): The x data points.
        y_data (numpy.ndarray): The observed y data points.
        true_params (dict): The true parameters used to generate the data.
        estimated_params (dict): The parameters estimated by ZeroGuess.
        test_name (str): Name of the test for file naming.
        monkeypatch (pytest.MonkeyPatch, optional): MonkeyPatch fixture for mocking plt.show().

    Returns:
        tuple: A tuple containing the three figure objects (fit_comparison, parameter_comparison, training_history).
    """
    # Mock plt.show() if monkeypatch is provided
    if monkeypatch:
        monkeypatch.setattr(plt, "show", lambda: None)

    # Ensure the output directory exists
    ensure_output_dir_exists()

    # Generate a timestamp for unique filenames
    timestamp = get_timestamp()

    # Create some fitted parameters (typically from optimization, but we'll use true params with small noise)
    # This simulates what would happen after further refinement with a curve fitting library
    np.random.seed(42)
    fitted_params = {
        param: value + np.random.normal(0, 0.05) for param, value in true_params.items()
    }

    # Test 1: Plot fit comparison
    fig1 = plot_fit_comparison(
        function=function,
        x_data=x_data,
        y_data=y_data,
        true_params=true_params,
        estimated_params=estimated_params,
        fitted_params=fitted_params,
    )
    assert fig1 is not None, "plot_fit_comparison should return a matplotlib figure"

    # Save the fit comparison figure
    fit_comparison_path = os.path.join(
        VISUALIZATION_OUTPUT_DIR, f"{test_name}_fit_comparison_{timestamp}.png"
    )
    fig1.savefig(fit_comparison_path, dpi=300, bbox_inches="tight")
    print(f"Saved fit comparison visualization to: {fit_comparison_path}")

    # Test 2: Plot parameter comparison
    fig2 = plot_parameter_comparison(
        true_params=true_params,
        estimated_params=estimated_params,
        fitted_params=fitted_params,
    )
    assert (
        fig2 is not None
    ), "plot_parameter_comparison should return a matplotlib figure"

    # Save the parameter comparison figure
    param_comparison_path = os.path.join(
        VISUALIZATION_OUTPUT_DIR, f"{test_name}_parameter_comparison_{timestamp}.png"
    )
    fig2.savefig(param_comparison_path, dpi=300, bbox_inches="tight")
    print(f"Saved parameter comparison visualization to: {param_comparison_path}")

    # We don't have training history in this utility function, so return None for fig3
    return fig1, fig2, None


def evaluate_prediction_quality(
    function,
    x_data,
    y_data,
    true_params,
    predicted_params,
    max_param_error=2.0,
    max_rmse=5.0,
    min_correlation=0.5,
):
    """Evaluate the quality of parameter estimation and curve fitting.

    Args:
        function (callable): The function used for curve fitting.
        x_data (numpy.ndarray): The x data points.
        y_data (numpy.ndarray): The observed y data points.
        true_params (dict): The true parameters used to generate the data.
        predicted_params (dict): The parameters predicted by ZeroGuess.
        max_param_error (float, optional): Maximum allowed relative error for parameters.
        max_rmse (float, optional): Maximum allowed RMSE for the fit.
        min_correlation (float, optional): Minimum required correlation between true and predicted curves.

    Returns:
        dict: A dictionary with evaluation metrics (param_errors, rmse, correlation).
    """
    from ..test_utils import calculate_parameter_error

    # Calculate errors compared to true parameters
    errors = calculate_parameter_error(predicted_params, true_params)

    # Check if errors are within tolerance
    for param_name, error in errors.items():
        assert error <= max_param_error, f"Error for {param_name} too high: {error:.2f}"

    # Use the predicted parameters to generate curve values
    y_predicted = function(x_data, **predicted_params)

    # Calculate RMSE between predicted curve and actual data
    residuals = y_data - y_predicted
    rmse = np.sqrt(np.mean(residuals**2))

    # RMSE should be reasonably small for a good prediction
    assert rmse < max_rmse, f"RMSE too high: {rmse:.2f}"

    # Generate clean data with predicted parameters
    y_clean_predicted = function(x_data, **predicted_params)

    # Generate clean data with true parameters
    y_clean_true = function(x_data, **true_params)

    # Calculate correlation between predicted and true clean signals
    correlation = np.corrcoef(y_clean_predicted, y_clean_true)[0, 1]

    # Correlation should be high
    assert correlation > min_correlation, f"Correlation too low: {correlation:.2f}"

    # Return the evaluation metrics
    return {"param_errors": errors, "rmse": rmse, "correlation": correlation}
