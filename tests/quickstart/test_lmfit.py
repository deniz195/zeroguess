import os

import lmfit
import matplotlib.pyplot as plt

from zeroguess.utils.visualization import plot_fit_comparison


def test_zeroguess_lmfit_quickstart(tmp_path):
    """Test ZeroGuess integration with lmfit using the quickstart workflow."""

    import numpy as np

    # Define a simple wavelet function directly
    def wavelet(x, frequency, phase, position, width):
        envelope = np.exp(-((x - position) ** 2) / (2 * width**2))
        return envelope * np.sin(2 * np.pi * frequency * (x - position) + phase)

    # Create some synthetic experimental data with known parameters
    true_params = {"frequency": 0.5, "phase": 1.0, "position": 7.0, "width": 1.5}

    # Generate x, y data points
    x_data = np.linspace(0, 20, 200)
    y_clean = wavelet(x_data, **true_params)

    # Add noise
    np.random.seed(42)  # For reproducibility
    noise_level = 0.05
    y_data = y_clean + np.random.normal(0, noise_level * (np.max(y_clean) - np.min(y_clean)), size=y_clean.shape)

    from zeroguess.integration import ZeroGuessModel

    # Enhanced lmfit Model with parameter estimation
    model = ZeroGuessModel(
        wavelet,
        independent_vars_sampling={"x": x_data},
        estimator_settings={
            # Configure training parameters
            # "n_samples": 1000,
            # "n_epochs": 200,
            # "validation_split": 0.2,
            # "add_noise": True,
            # "noise_level": 0.1,
            # 'verbose': True
            # Provide a function to make parameters canonical
            # "make_canonical": ...,
            # Save and load model automatically
            "snapshot_path": "estimator_lmfit.pth",
        },
    )

    # Set parameter hints
    model.set_param_hint("frequency", min=0.05, max=1.0)
    model.set_param_hint("phase", min=0.0, max=2.0 * np.pi)
    model.set_param_hint("position", min=5.0, max=15.0)
    model.set_param_hint("width", min=0.1, max=3.0)

    # Guess parameters with ZeroGuess estimator
    params = model.guess(y_data, x=x_data)

    # Run the fit
    result = model.fit(y_data, x=x_data, params=params)

    # Run verifications

    # Verify that guess produces a valid lmfit Parameters object
    assert isinstance(params, lmfit.Parameters)
    assert set(params.keys()) == set(true_params.keys())

    # Extract initial parameter values
    initial_params = {name: param.value for name, param in params.items()}

    # Verify initial parameters are within expected bounds
    assert 0.05 <= initial_params["frequency"] <= 1.0
    assert 0.0 <= initial_params["phase"] <= 2.0 * np.pi
    assert 5.0 <= initial_params["position"] <= 15.0
    assert 0.1 <= initial_params["width"] <= 3.0

    # Verify fit result is valid
    assert isinstance(result, lmfit.model.ModelResult)
    assert result.success

    # Extract fitted parameter values
    fitted_params = {name: param.value for name, param in result.params.items()}

    # Verify fit parameters are close to true values (within reasonable tolerance)
    # Use higher tolerance for noisy data
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
    assert os.path.exists(model.estimator_settings["snapshot_path"])

    # Create plot output path in the temporary directory
    plot_path = os.path.join(tmp_path, "fit_comparison.png")
    print(f"Plot path: {plot_path}")

    # Generate visualization for documentation
    fig = plot_fit_comparison(
        function=wavelet,
        x_data=x_data,
        y_data=y_data,
        true_params=true_params,
        estimated_params=initial_params,
        fitted_params=fitted_params,
        title="Wavelet Function Fit Comparison",
    )

    # Save figure to temporary directory
    fig.savefig(plot_path)
    plt.close(fig)

    # Verify plot was created
    assert os.path.exists(plot_path)

    return initial_params, fitted_params
