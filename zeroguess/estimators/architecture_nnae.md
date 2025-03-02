# Neural Network Autoencoder Estimator (NNAE) Architecture

## Overview

The Neural Network Autoencoder Estimator (NNAE) represents an evolution over the existing neural network estimator in ZeroGuess. It leverages an autoencoder architecture to improve parameter estimation accuracy and robustness, especially for complex curve fitting problems where parameter identification may be ambiguous.

## Architecture Design

### Conceptual Framework

The NNAE architecture consists of two main components:

1. **Encoder Network**: Maps from function observations (y-values at sampling points) to parameter estimates.
2. **Decoder Network**: Maps from parameter estimates to the first 10 central moments of the function, rather than reconstructing the full function values directly.

This design creates a more efficient learning path by focusing on capturing the statistical essence of the function rather than its exact shape:

```
Input (y-values) → Encoder → Parameters → Decoder → Function Moments
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
- Outputs the first 10 central moments of the function, capturing its statistical properties
- This approach reduces dimensionality and focuses on essential features rather than precise reconstruction
- Structure typically follows an MLP architecture with specific layers designed for moment calculation

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

### Three-Step Training Process

The NNAE estimator employs a specialized three-step training process that targets each component individually before fine-tuning the entire model:

1. **Step 1: Encoder Training**
   - Train only the encoder network to map function observations to parameters
   - Freeze the decoder weights during this phase
   - Use parameter estimation loss (matching ground truth parameters)
   - Typically runs for 1/3 of the total training epochs

2. **Step 2: Decoder Training**
   - Train only the decoder network to map parameters to function moments
   - Freeze the encoder weights during this phase
   - Use moment reconstruction loss (matching computed moments from function values)
   - Typically runs for 1/3 of the total training epochs

3. **Step 3: End-to-End Training**
   - Train the entire network with all components connected
   - Use the full tri-component loss function described below
   - Fine-tunes the connections between encoder and decoder
   - Typically runs for the remaining 1/3 of training epochs

This staged approach ensures that each component learns its specific task effectively before the entire system is optimized together.

### Tri-Component Loss Training

The NNAE architecture employs a specialized tri-component loss function:

1. **Moment Reconstruction Loss** (L₁): Measures the difference between the moments of input function and the moments predicted by the decoder.
   - L₁ = MSE(moments(y_input), moments_predicted)
   - Where moments_predicted = decoder(encoder(y_input))

2. **Parameter Validation Loss** (L₂): Measures the difference between input function points and function points computed using the estimated parameters.
   - L₂ = MSE(y_input, f(x, θ))
   - Where θ = encoder(y_input) and f is the original fit function

3. **Parameter Accuracy Loss** (L₃): Directly measures the accuracy of parameter estimation (used during training with synthetic data where true parameters are known).
   - L₃ = MSE(θ_true, θ_predicted)
   - Where θ_predicted = encoder(y_input)

4. **Combined Loss**: L = α·L₁·L₂ + β·L₃
   - This formula balances both reconstruction quality and parameter accuracy
   - The multiplication of L₁ and L₂ creates a coupling effect that emphasizes both aspects simultaneously
   - Where α and β are weighting coefficients that balance the components
   - Default values: α = 0.3, β = 0.7 (prioritizing parameter validation loss)

### Training Process Details

1. Forward Pass:
   - Encoder processes input y-values to produce parameter estimates
   - Parameter normalization is applied
   - Decoder computes the first 10 central moments from parameters
   - Fit function calculates y-values from parameters (separate path)

2. Loss Calculation:
   - Calculate moment reconstruction loss between input moments and decoder output
   - Calculate parameter validation loss between input and fit function output
   - Calculate parameter accuracy loss between true and predicted parameters
   - Combine losses using the formula α·L₁·L₂ + β·L₃

3. Backward Pass:
   - Gradients flow through all components based on the active training step
   - During encoder training, only encoder weights are updated
   - During decoder training, only decoder weights are updated
   - During end-to-end training, all weights are updated based on the combined loss

### Training Advantages

1. **Moment-Based Learning**: By focusing on statistical moments rather than exact function values, the decoder learns to capture the essential characteristics of functions more efficiently.

2. **Staged Learning**: The three-step process allows each component to learn its specific task before integrating, reducing the risk of suboptimal convergence.

3. **Parameter Consistency**: By incorporating parameter accuracy directly into the loss function, the model prioritizes correct parameter estimation.

4. **Multi-objective Optimization**: By optimizing for moment reconstruction, function validation, and parameter accuracy simultaneously, the model develops more robust representations.

5. **Dimensionality Reduction**: Using moments reduces the dimensionality of the decoder's output space, making learning more efficient and focused on essential features.

## Implementation Details

The NNAE estimator is implemented as a separate estimator type alongside the existing NeuralNetworkEstimator, following a similar structure and API pattern for consistency.

### Implementation Strategy

1. **Dedicated Module**: Implemented in the `nnae_estimator.py` file in the `zeroguess/estimators/` directory
2. **Class Inheritance**: Inherits from the same `BaseEstimator` class as `NeuralNetworkEstimator`
3. **Consistent API**: Maintains the same public API for seamless interchangeability
4. **Enhanced Training Process**: Implements the three-step training process described above

### Architecture Support

For the initial implementation, NNAE supports the MLP (Multilayer Perceptron) architecture with enhancements:

1. **Encoder**: A series of fully connected layers that progressively reduce dimensionality
   - Input: Function observations (y-values at sampling points)
   - Output: Estimated parameters
   - Default structure: [128, 256, 256, 128, 64] hidden units

2. **Decoder**: An enhanced network with residual connections and increased capacity
   - Input: Estimated parameters
   - Output: First 10 central moments of the function
   - Default structure: [64, 128, 256, 256, 128] hidden units with residual connections
   - Includes batch normalization and skip connections to improve gradient flow

3. **Parameter Normalization**: A sigmoid activation layer between encoder and decoder
   - Ensures parameters are normalized to [0, 1] range
   - Parameters are denormalized to their original ranges when used

### Key Implementation Components

The implementation includes:

1. **NNAEEstimator Class**: Main estimator class that follows the BaseEstimator interface
2. **Three-Step Training Process**: Implementation of the specialized training methodology
3. **Tri-Component Loss Function**: Implementation of the combined loss with parameter accuracy component
4. **Moment Calculation**: Methods to compute and compare statistical moments from function values
5. **Enhanced Decoder**: Advanced decoder architecture with residual connections and increased capacity

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
        "encoder_layers": [128, 256, 256, 128, 64],  # Encoder hidden layer sizes
        "decoder_layers": [64, 128, 256, 256, 128],  # Decoder hidden layer sizes
        "activation": "relu",
        "dropout_rate": 0.1
    },
    alpha=0.3,  # Weight for moment reconstruction and parameter accuracy
    beta=0.7    # Weight for parameter validation loss
)

# Train with control over individual phases
estimator.train(
    n_samples=1000, 
    batch_size=32, 
    n_epochs=300,
    encoder_epochs=100,   # Custom epochs for encoder training
    decoder_epochs=100,   # Custom epochs for decoder training
    end_to_end_epochs=100 # Custom epochs for end-to-end training
)

# Use for parameter estimation
x_data = np.linspace(-10, 10, 100)
y_data = gaussian(x_data, 5.0, 1.0, 0.8) + np.random.normal(0, 0.1, 100)
params = estimator.predict(x=x_data, y=y_data)
```

## Advantages over Standard NN Estimator

1. **Statistical Focus**: By targeting moments rather than exact reconstruction, the model focuses on the most informative aspects of the function.

2. **Enhanced Training Stability**: The three-step training process prevents one component from dominating the learning process.

3. **More Efficient Learning**: The decoder's reduced output dimensionality (10 moments vs. full function) makes the learning task more tractable.

4. **Direct Parameter Optimization**: The inclusion of parameter accuracy in the loss function provides a more direct learning signal.

5. **Improved Generalization**: The combination of moment-based reconstruction and multi-step training leads to better generalization to unseen data.

6. **Reduced Sensitivity to Initialization**: The staged training approach makes the model less sensitive to weight initialization.

## Limitations and Considerations

1. **Increased Computational Cost**: The three-step training process and complex loss calculation increase training time compared to the standard NN estimator.

2. **Moment Calculation Overhead**: Computing statistical moments adds computational complexity during training.

3. **Loss Balancing**: Finding the optimal balance between the three loss components may require tuning.

4. **Memory Requirements**: The enhanced decoder architecture requires more memory due to the additional connections and layers.

## Future Extensions

1. **Adaptive Moment Selection**: Dynamically selecting which moments to focus on based on function characteristics.

2. **Variational Implementation**: Extending to a variational autoencoder (VAE) approach to better handle uncertainty in parameter estimation.

3. **Parameter Uncertainty Estimation**: Using the reconstruction error as a proxy for parameter uncertainty.

4. **Transfer Learning**: Pre-training on similar function families to improve performance on specific instances.

5. **Adaptive Loss Weighting**: Dynamically adjusting the weights of loss components during training based on their relative magnitudes and gradients.




