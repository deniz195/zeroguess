import numpy as np

from zeroguess.functions import DoubleGaussianFunction, add_gaussian_noise
from zeroguess.integration import lmfit_integration

# Create a Gaussian function
double_gaussian = DoubleGaussianFunction()

# Create some experimental data
true_params = double_gaussian.get_random_params()
x_data = np.linspace(-10, 10, 100)
y_data = add_gaussian_noise(double_gaussian(x_data, **true_params), sigma=0.1)

# Enhanced lmfit Model with parameter estimatio
model = lmfit_integration.Model(
    double_gaussian,
    independent_vars_sampling={"x": np.linspace(-10, 10, 100)},
    estimator_settings={
        "make_canonical": double_gaussian.get_canonical_params,
        "n_samples": 1000,
        "n_epochs": 200,
        "validation_split": 0.2,
        "add_noise": True,
        "noise_level": 0.1,
        # 'verbose': True
        "snapshot_path": "model_dg.pth"
    },
)

model.set_param_hint("amplitude1", min=0, max=10)
model.set_param_hint("center1", min=-5, max=5)
model.set_param_hint("width1", min=0.1, max=2)
model.set_param_hint("amplitude2", min=0, max=10)
model.set_param_hint("center2", min=-5, max=5)
model.set_param_hint("width2", min=0.1, max=2)

# Standard lmfit workflow
params = model.guess(y_data, x=x_data)

result = model.fit(y_data, x=x_data, params=params)
print(result.fit_report())
