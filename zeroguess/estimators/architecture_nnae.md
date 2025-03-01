# Neural Network Autoencoder Estimator (NNAE) Architecture

## Overview

The Neural Network Autoencoder Estimator (NNAE) represents an evolution over the existing neural network estimator in ZeroGuess. It leverages an autoencoder architecture to improve parameter estimation accuracy and robustness, especially for complex curve fitting problems where parameter identification may be ambiguous.

## Architecture Design

### Conceptual Framework

The NNAE architecture consists of two main components:

1. **Encoder Network**: Maps from function observations (y-values at sampling points) to parameter estimates.
2. **Decoder Network**: Maps from parameter estimates back to function observations.

This design creates a complete input-to-output-to-input cycle, providing additional constraints during training that improve the quality of parameter estimates.

```
Input (y-values) → Encoder → Parameters → Decoder → Reconstructed y-values
                              ↓
                     Fit Function Evaluation
                              ↓
                     Computed y-values
```

### Key Components

#### Encoder Network
- Functionally identical to the current neural network estimator
- Takes sampled fit function points (y-values) as input
- Outputs estimated parameter values
- Structure typically follows an MLP architecture with decreasing layer sizes

#### Decoder Network
- Takes the estimated parameter values as input
- Outputs reconstructed function points (y-values)
- Structure typically follows an MLP architecture with increasing layer sizes
- Mirrors the encoder architecture in reverse

#### Parameter Normalization Layer
- Sits between encoder and decoder
- Ensures parameter values remain within defined ranges
- Implements differentiable clipping or activation functions

#### Fit Function Integration Layer
- Non-trainable computational layer
- Takes estimated parameters from encoder
- Computes function values using the original fit function
- Used only for loss calculation, not in the forward pass

## Training Methodology

### Dual-Loss Training

The NNAE architecture employs a specialized dual-loss function:

1. **Reconstruction Loss** (L₁): Measures the difference between input function points and reconstructed function points from the decoder.
   - L₁ = MSE(y_input, y_reconstructed)
   - Where y_reconstructed = decoder(encoder(y_input))

2. **Parameter Validation Loss** (L₂): Measures the difference between input function points and function points computed using the estimated parameters.
   - L₂ = MSE(y_input, f(x, θ))
   - Where θ = encoder(y_input) and f is the original fit function

3. **Combined Loss**: L = α·L₁ + β·L₂
   - Where α and β are weighting coefficients that balance the two loss components
   - Default values: α = 0.3, β = 0.7 (prioritizing parameter validation loss)

### Training Process

1. Forward Pass:
   - Encoder processes input y-values to produce parameter estimates
   - Parameter normalization is applied
   - Decoder reconstructs y-values from parameters
   - Fit function calculates y-values from parameters (separate path)

2. Loss Calculation:
   - Calculate reconstruction loss between input and decoder output
   - Calculate parameter validation loss between input and fit function output
   - Combine losses using weighting coefficients

3. Backward Pass:
   - Gradients flow through both decoder and parameter validation paths
   - Encoder weights are updated based on combined gradients
   - Decoder weights are updated based only on reconstruction loss gradients

### Training Advantages

1. **Parameter Consistency**: By forcing the model to reconstruct the input data, the encoder must learn parameter representations that contain enough information to reproduce the original function.

2. **Regularization Effect**: The reconstruction task serves as a regularizer, preventing the model from overfitting to the specific parameter estimation task.

3. **Multi-objective Optimization**: By optimizing for both direct parameter estimation and function reconstruction, the model develops more robust representations.

4. **Ambiguity Resolution**: For functions where multiple parameter sets produce similar outputs, the autoencoder architecture can help identify more stable parameter regions.

## Implementation Details

The NNAE estimator will be implemented as a separate estimator type alongside the existing NeuralNetworkEstimator, following a similar structure and API pattern for consistency.

### Implementation Strategy

1. **New Module File**: Create a dedicated `nnae_estimator.py` file in the `zeroguess/estimators/` directory
2. **Class Inheritance**: Inherit from the same `BaseEstimator` class as `NeuralNetworkEstimator`
3. **Consistent API**: Maintain the same public API for seamless interchangeability
4. **Architecture Parameterization**: Use the same architecture and architecture_params pattern as NeuralNetworkEstimator

### Architecture Support

For the initial implementation, NNAE will only support the MLP (Multilayer Perceptron) architecture. The architecture parameter will be accepted for API consistency with NeuralNetworkEstimator, but only the "mlp" option will be fully implemented. Other architecture types like "cnn" and "transformer" are planned for future extensions.

The MLP architecture for NNAE consists of:

1. **Encoder**: A series of fully connected layers that progressively reduce dimensionality
   - Input: Function observations (y-values at sampling points)
   - Output: Estimated parameters
   - Default structure: [64, 32] hidden units

2. **Decoder**: A series of fully connected layers that progressively increase dimensionality
   - Input: Estimated parameters
   - Output: Reconstructed function observations
   - Default structure: [32, 64] hidden units

3. **Parameter Normalization**: A sigmoid activation layer between encoder and decoder
   - Ensures parameters are normalized to [0, 1] range
   - Parameters are denormalized to their original ranges when used

### Key Implementation Components

The implementation will include:

1. **NNAEEstimator Class**: Main estimator class that follows the BaseEstimator interface
2. **Dual-Loss Training**: Implementation of the specialized loss function that combines reconstruction and parameter validation losses
3. **Network Architecture**: MLP-based autoencoder with separate encoder and decoder components
4. **Factory Integration**: Registration with the estimator factory for seamless creation

### Usage Example

```python
import zeroguess
import numpy as np

# Define function to fit
def gaussian(x, amplitude, center, width):
    return amplitude * np.exp(-(x - center)**2 / (2 * width**2))

# Define sampling points
x_sampling = np.linspace(-10, 10, 100)

# Create NNAE estimator
estimator = zeroguess.create_estimator(
    function=gaussian,
    param_ranges={
        'amplitude': (0, 10),
        'center': (-5, 5),
        'width': (0.1, 2)
    },
    independent_vars_sampling={'x': x_sampling},
    estimator_type="nnae",  # Select NNAE estimator type
    architecture="mlp",     # Only MLP is supported for now
    architecture_params={
        "encoder_layers": [64, 32],  # Encoder hidden layer sizes
        "decoder_layers": [32, 64],  # Decoder hidden layer sizes
        "activation": "relu",
        "dropout_rate": 0.1
    },
    alpha=0.3,  # Reconstruction loss weight
    beta=0.7    # Parameter validation loss weight
)

# Train and use as normal
estimator.train(n_samples=1000, batch_size=64, n_epochs=100)

# Use for parameter estimation
x_data = np.linspace(-10, 10, 100)
y_data = gaussian(x_data, 5.0, 1.0, 0.8) + np.random.normal(0, 0.1, 100)
params = estimator.predict(x=x_data, y=y_data)
```

## Advantages over Standard NN Estimator

1. **Improved Generalization**: The dual-loss approach helps the model generalize better to unseen data.

2. **Better Parameter Stability**: The reconstruction loss provides an additional regularizing constraint that leads to more stable parameter estimates.

3. **Enhanced Performance on Ambiguous Functions**: For functions where multiple parameter sets can produce similar outputs, the NNAE architecture can better identify the most likely parameter region.

4. **Interpretable Intermediate Outputs**: The decoder can be used to visualize the function corresponding to the estimated parameters, providing a check on estimation quality.

5. **Reduced Sensitivity to Initialization**: The dual-loss training regime makes the model less sensitive to weight initialization.

## Limitations and Considerations

1. **Increased Computational Cost**: The additional decoder network and dual-loss calculation increase training time compared to the standard NN estimator.

2. **Function Differentiability**: The parameter validation loss requires the fit function to be differentiable with respect to its parameters for optimal training.

3. **Loss Balancing**: Finding the optimal balance between reconstruction loss and parameter validation loss may require tuning.

4. **Memory Requirements**: The NNAE architecture requires more memory due to the additional decoder network.

## Future Extensions

1. **Variational Implementation**: Extending to a variational autoencoder (VAE) approach to better handle uncertainty in parameter estimation.

2. **Parameter Uncertainty Estimation**: Using the reconstruction error as a proxy for parameter uncertainty.

3. **Transfer Learning**: Pre-training on similar function families to improve performance on specific instances.

4. **Specialized Architectures**: Creating specialized NNAE architectures for common function types (peaks, oscillations, etc.).




