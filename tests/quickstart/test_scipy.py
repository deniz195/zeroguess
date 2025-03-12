import os

import matplotlib.pyplot as plt

from zeroguess.utils.visualization import plot_fit_comparison


def test_zeroguess_scipy_quickstart(tmp_path):
    """Test ZeroGuess integration with scipy using the quickstart workflow."""

    import numpy as np

    # Define a simple wavelet function directly
    def wavelet(x, frequency, phase, position, width):
        z = (x - position) / width
        return np.exp(-(z**2)) * np.cos(2 * np.pi * frequency * z + phase)

    # Create some synthetic experimental data with known parameters
    true_params = {"frequency": 0.5, "phase": 1.0, "position": 7.0, "width": 1.5}

    # Generate x, y data points
    x_data = np.linspace(0, 20, 200)
    y_clean = wavelet(x_data, **true_params)

    # Add noise
    np.random.seed(42)  # For reproducibility
    noise_level = 0.05
    y_data = y_clean + np.random.normal(0, noise_level * (np.max(y_clean) - np.min(y_clean)), size=y_clean.shape)

    from scipy import optimize

    import zeroguess

    # Create and train parameter estimator
    estimator = zeroguess.create_estimator(
        function=wavelet,
        param_ranges={
            "frequency": (0.05, 1.0),
            "phase": (0.0, 2.0 * np.pi),
            "position": (5.0, 15.0),
            "width": (0.1, 3.0),
        },
        independent_vars_sampling={"x": x_data},
        snapshot_path="model_dg_plain.pth",  # saves and loads model automatically
    )

    if not estimator.is_trained:
        estimator.train()

    # Get parameter estimates for experimental data
    initial_params = estimator.predict(x_data, y_data)

    # Use in standard curve fitting
    optimal_params, pcov = optimize.curve_fit(wavelet, x_data, y_data, p0=list(initial_params.values()))

    # Run verifications

    # Verify that predict produces valid parameters
    assert set(initial_params.keys()) == set(true_params.keys())

    # Verify initial parameters are within expected bounds
    assert 0.05 <= initial_params["frequency"] <= 1.0
    assert 0.0 <= initial_params["phase"] <= 2.0 * np.pi
    assert 5.0 <= initial_params["position"] <= 15.0
    assert 0.1 <= initial_params["width"] <= 3.0

    # Create a dictionary of fitted parameters (SciPy curve_fit returns values in order of function parameters)
    param_names = list(true_params.keys())
    fitted_params = {name: value for name, value in zip(param_names, optimal_params)}

    # Verify fit parameters are close to true values (within reasonable tolerance)
    assert abs(fitted_params["frequency"] - true_params["frequency"]) < 0.2
    assert abs(fitted_params["position"] - true_params["position"]) < 1.0
    assert abs(fitted_params["width"] - true_params["width"]) < 0.5

    # Phase is periodic, so we need to handle it differently
    phase_diff = min(
        abs(fitted_params["phase"] - true_params["phase"]),
        abs(fitted_params["phase"] - true_params["phase"] - 2 * np.pi),
        abs(fitted_params["phase"] - true_params["phase"] + 2 * np.pi),
    )
    assert phase_diff < 1.0

    # Verify the model file was created
    assert os.path.exists(estimator.snapshot_path)

    # Create plot output path in the temporary directory
    plot_path = os.path.join(tmp_path, "scipy_fit_comparison.png")

    # Generate visualization for documentation
    fig = plot_fit_comparison(
        function=wavelet,
        x_data=x_data,
        y_data=y_data,
        true_params=true_params,
        estimated_params=initial_params,
        fitted_params=fitted_params,
        title="SciPy Wavelet Function Fit Comparison",
    )

    # Save figure to temporary directory
    fig.savefig(plot_path)
    plt.close(fig)

    # Verify plot was created
    assert os.path.exists(plot_path)

    # Calculate fit quality metrics
    residuals = y_data - wavelet(x_data, **fitted_params)
    mse = np.mean(residuals**2)

    # Log fit quality metrics for debugging
    print(f"Mean Squared Error: {mse:.6f}")
    print(f"Initial parameters: {initial_params}")
    print(f"Fitted parameters: {fitted_params}")
    print(f"True parameters: {true_params}")

    # Return results for potential further analysis
    return initial_params, fitted_params, mse
