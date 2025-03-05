import numpy as np

from zeroguess.functions import GaussianFunction, add_gaussian_noise
from zeroguess.integration import lmfit_integration

# Create a Gaussian function
gaussian = GaussianFunction()

# Create some experimental data
true_params = gaussian.get_random_params()
x_data = np.linspace(-10, 10, 100)
y_data = add_gaussian_noise(gaussian(x_data, **true_params), sigma=0.1)

# Enhanced lmfit Model with parameter estimatio
model = lmfit_integration.Model(gaussian, independent_vars_sampling={"x": np.linspace(-10, 10, 100)})

model.set_param_hint("amplitude", min=0, max=10)
model.set_param_hint("center", min=-5, max=5)
model.set_param_hint("width", min=0.1, max=2)

# Standard lmfit workflow
params = model.guess(y_data, x=x_data)

result = model.fit(y_data, x=x_data, params=params)
print(result.fit_report())
