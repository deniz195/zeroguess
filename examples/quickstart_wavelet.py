import numpy as np

from zeroguess.functions import WaveletFunction, add_gaussian_noise
from zeroguess.integration import ZeroGuessModel

# Create a Gaussian function
wavelet = WaveletFunction()

# Create some experimental data
true_params = wavelet.get_random_params()
x_data = np.linspace(-10, 10, 100)
y_data = add_gaussian_noise(wavelet(x_data, **true_params), sigma=0.1)

# Enhanced lmfit Model with parameter estimatio
model = ZeroGuessModel(
    wavelet,
    independent_vars_sampling={"x": np.linspace(-10, 10, 100)},
    estimator_settings={
        "make_canonical": wavelet.get_canonical_params,
        "n_samples": 1000,
        "n_epochs": 200,
        "validation_split": 0.2,
        "add_noise": True,
        "noise_level": 0.1,
        # 'verbose': True
    },
)

model.set_param_hint("frequency", min=0.05, max=1.0)
model.set_param_hint("phase", min=0.0, max=2.0 * np.pi)
model.set_param_hint("position", min=5.0, max=15.0)
model.set_param_hint("width", min=0.1, max=3.0)

# Standard lmfit workflow
params = model.guess(y_data, x=x_data)

result = model.fit(y_data, x=x_data, params=params)
print(result.fit_report())
