# ZeroGuess: Machine Learning for Curve Fitting Parameter Estimation

## What
We are building a Python library (ZeroGuess) that simplifies the estimation of starting parameters for curve fitting. The library supports SciPy and lmfit, two widely used curve fitting libraries in the scientific Python ecosystem.

## Why
While curve fitting is a well-understood problem, the process of estimating starting parameters is not. It is a very tedious and error-prone process that often requires domain expertise, trial and error, or both. Poor initial parameter estimates can cause fitting algorithms to:
- Converge to suboptimal local minima
- Require more iterations to converge
- Fail to converge entirely

Machine learning can utilize knowledge of the curve fitting target function to learn a function that estimates optimal starting parameters, saving time and improving results for scientists and engineers.

## How
The user specifies:
1. The curve fitting target function
2. Relevant ranges for the starting parameters
3. Sampling points for independent variables

The library then:
1. Generates a random dataset of starting parameters within the specified ranges
2. Calculates the corresponding curve fitting target function values at the sampling points
3. Uses this synthetic dataset to train a neural network model (implemented in PyTorch)
4. The trained model can then predict optimal starting parameters for new experimental data

The neural network architecture will be based on an autoencoder-like approach, where the encoder is given by the fitting function and the decoder is learned. The exact architecture will be determined during implementation.

## What are the inputs?
- Curve fitting target function (compatible with SciPy or lmfit)
- Relevant range for each starting parameter (min/max values)
- Sampling points for independent variables
- (Optional) Training configuration parameters (epochs, learning rate, etc.)
- (Optional) Dataset with noise characteristics for more realistic training

## What are the outputs?
- Starting parameter estimates for the specified function
- Confidence metrics for the estimates
- Diagnostic information about the training process
- Warnings for potential issues (ambiguities, noise levels, etc.)

## Technical Requirements

### Performance Metrics
- For numerically well-behaved models, parameter estimates should be within 5% of optimal values when presented with datasets having SNR ≥ 10 (i.e., noise < 0.1 × signal)
- Training time should be reasonable for typical functions (minutes, not hours)
- Inference time should be fast (<1s) to allow interactive use

### API Design
- The library should seamlessly integrate with supported curve fitting libraries
- It should provide a starting parameter estimation function requiring minimal additional information beyond what is already specified within the library APIs
- It should follow consistent naming conventions and patterns familiar to users of SciPy and lmfit

### Error Handling
The library should properly handle edge cases:
- Functions difficult to fit: Issue a warning that the function is difficult to fit
- Noisy data: Issue a warning that the data is out of distribution for the learned model
- Functions with multiple local minima: Issue a warning that the function has ambiguities and might not be able to be learned correctly
- Cases where learning fails to converge: Issue an error that learning was not successful

### Compatibility Requirements
- Python 3.10 or higher
- Platform-independent implementation
- Minimal dependencies beyond PyTorch
- Compatible with both SciPy and lmfit APIs

## Success Criteria
The library will be considered successful if:
1. It demonstrably reduces the time and effort required to find good starting parameters
2. It achieves the target performance metrics for a set of common fitting functions
3. It integrates smoothly with existing workflows
4. It provides clear guidance when limitations are encountered

## Future Considerations
- Extending support to additional curve fitting libraries
- Implementing more sophisticated machine learning architectures
- Adding support for constraints and boundary conditions
- Creating a web service for cloud-based parameter estimation


