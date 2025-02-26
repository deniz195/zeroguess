# ZeroGuess Examples

This directory contains example scripts that demonstrate how to use the ZeroGuess library for parameter estimation in curve fitting.

## Architecture Selection Example

The `architecture_selection_example.py` script demonstrates how to use different neural network architectures for parameter estimation.

### Usage

```bash
# Run with default MLP architecture
python architecture_selection_example.py --train --predict

# Run with specific architecture (e.g., MLP)
python architecture_selection_example.py --architecture mlp --train --predict

# Specify custom architecture parameters
python architecture_selection_example.py --architecture mlp --architecture-params "hidden_layers:[64,128,64] dropout_rate:0.2" --train --predict

# Run with more training samples and epochs
python architecture_selection_example.py --architecture mlp --samples 2000 --epochs 200 --train --predict
```

### Command Line Arguments

- `--architecture`: Neural network architecture to use (choices: "mlp", "cnn", "transformer", "best"; default: "mlp")
- `--train`: Train a new model (otherwise tries to load a pre-trained model)
- `--predict`: Perform prediction using the model
- `--samples`: Number of training samples to generate (default: 1000)
- `--epochs`: Number of training epochs (default: 100)
- `--batch-size`: Batch size for training (default: 64)
- `--architecture-params`: Architecture-specific parameters in the format "param:value param2:value2"

### Currently Supported Architectures

- **MLP (Multilayer Perceptron)**: The default architecture, fully implemented.
- **CNN (Convolutional Neural Network)**: Optimized for oscillatory and multi-peak functions.
- **Transformer**: Architecture based on self-attention mechanisms, planned for future work.

You can also use "best" to automatically select the recommended architecture for your problem.

## CNN Architecture Test Example

The `cnn_architecture_test.py` script demonstrates the CNN architecture for parameter estimation of oscillatory functions like damped sine waves.

### Usage

```bash
# Train and test CNN architecture
python cnn_architecture_test.py --train --predict

# Compare with MLP architecture
python cnn_architecture_test.py --architecture mlp --train --predict

# Customize CNN architecture parameters
python cnn_architecture_test.py --architecture-params "n_conv_layers:4 filters:[16,32,64,128] kernel_size:7" --train --predict

# Run with more training data
python cnn_architecture_test.py --samples 2000 --epochs 200 --train --predict
```

### Function

The test script uses a damped sine wave function with the following parameters:
- `amplitude`: Peak amplitude
- `frequency`: Oscillation frequency
- `phase`: Phase offset
- `decay`: Exponential decay rate

This type of oscillatory function is well-suited for CNN architecture's pattern detection capabilities.

## Notes

- The Transformer architecture is a placeholder for future work and will raise `NotImplementedError` if selected.
- The example always runs in the current directory, writing output files (plots and model files) to this location.
- If you specify `--predict` without `--train`, the script will try to load a previously saved model file. If it cannot find one, it will train a new model automatically. 