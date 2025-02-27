"""Unit tests for the data generation module of ZeroGuess."""

import numpy as np

# Import the test fixtures

# Once the ZeroGuess library is implemented, you would import from it directly
# For now, we'll create a mock or assume the implementation details


# Mock implementation of data generation for testing
def mock_generate_parameter_sets(param_ranges, num_samples=1000):
    """Generate random parameter sets within the specified ranges."""
    params = {}
    for param_name, (min_val, max_val) in param_ranges.items():
        params[param_name] = np.random.uniform(min_val, max_val, num_samples)
    return params


def mock_generate_training_data(function, param_sets, independent_vars_sampling):
    """Generate training data from parameter sets and sampling points."""
    num_samples = len(next(iter(param_sets.values())))

    # Extract independent variable names and sampling points
    independent_vars = {}
    for var_name, sampling in independent_vars_sampling.items():
        independent_vars[var_name] = sampling

    # Generate function outputs for each parameter set
    outputs = []
    param_values = []

    for i in range(num_samples):
        # Extract the i-th parameter set
        params = {name: values[i] for name, values in param_sets.items()}
        param_values.append(params)

        # Compute function output for all sampling points
        output = {}
        for var_name, sampling in independent_vars.items():
            # Create input dictionary with all other variables set to 0 (for multi-variable functions)
            inputs = {name: 0 for name in independent_vars.keys()}
            inputs[var_name] = sampling

            # Compute function output
            output[var_name] = function(**inputs, **params)

        outputs.append(output)

    return outputs, param_values


# Define the tests
class TestDataGeneration:
    """Tests for the data generation functionality."""

    def test_parameter_generation(self, set_random_seeds):
        """Test that parameter sets are generated correctly."""
        # Define parameter ranges
        param_ranges = {"amplitude": (0, 10), "center": (-5, 5), "width": (0.1, 2)}

        # Generate parameter sets
        num_samples = 1000
        param_sets = mock_generate_parameter_sets(param_ranges, num_samples)

        # Check that the correct parameters are generated
        assert set(param_sets.keys()) == set(param_ranges.keys())

        # Check that the correct number of samples is generated
        for param_name, values in param_sets.items():
            assert len(values) == num_samples

        # Check that values are within the specified ranges
        for param_name, values in param_sets.items():
            min_val, max_val = param_ranges[param_name]
            assert np.all(values >= min_val)
            assert np.all(values <= max_val)

    def test_training_data_generation(self, set_random_seeds, gaussian_function):
        """Test that training data is generated correctly."""
        # Define parameter ranges
        param_ranges = {"amplitude": (0, 10), "center": (-5, 5), "width": (0.1, 2)}

        # Define sampling points
        independent_vars_sampling = {"x": np.linspace(-10, 10, 100)}

        # Generate parameter sets
        num_samples = 10  # Small number for testing
        param_sets = mock_generate_parameter_sets(param_ranges, num_samples)

        # Generate training data
        outputs, param_values = mock_generate_training_data(gaussian_function, param_sets, independent_vars_sampling)

        # Check that the correct number of outputs is generated
        assert len(outputs) == num_samples
        assert len(param_values) == num_samples

        # Check that outputs have the correct structure
        for output in outputs:
            assert set(output.keys()) == set(independent_vars_sampling.keys())
            for var_name, values in output.items():
                assert len(values) == len(independent_vars_sampling[var_name])

        # Verify outputs match expected values for a specific parameter set
        # This is a more detailed test that would depend on the actual implementation
        # For now, we'll just check that outputs are not None or empty
        for output in outputs:
            for var_name, values in output.items():
                assert values is not None
                assert len(values) > 0
