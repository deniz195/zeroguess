"""Unit tests for the data generation module of ZeroGuess."""

from typing import Dict

import numpy as np
import pytest

from zeroguess.data.generators import SyntheticDataGenerator


# Define test functions
def gaussian_function(x, amplitude, center, width):
    """Simple Gaussian function for testing."""
    return amplitude * np.exp(-((x - center) ** 2) / (2 * width**2))


def gaussian_test_func(x, a, b, c):
    """Simple test function for the generator."""
    return a * np.exp(-((x - b) ** 2) / (2 * c**2))


def make_canonical_test(params: Dict[str, float]) -> Dict[str, float]:
    """Test canonical transformation function."""
    result = params.copy()
    # Simple transformation: ensure c is positive
    result["c"] = abs(result["c"])
    # Sort peaks by position if multiple peaks
    if "b1" in result and "b2" in result and result["b1"] > result["b2"]:
        result["a1"], result["a2"] = result["a2"], result["a1"]
        result["b1"], result["b2"] = result["b2"], result["b1"]
        result["c1"], result["c2"] = result["c2"], result["c1"]
    return result


# Mock implementations from original test file
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


# Define fixtures
@pytest.fixture
def set_random_seeds():
    """Set random seeds for reproducibility."""
    np.random.seed(42)
    return None


@pytest.fixture
def gaussian_function_fixture():
    """Return a simple Gaussian function for testing."""
    return gaussian_function


# Tests from the original data_generation file
class TestMockDataGeneration:
    """Tests for the legacy mock data generation functionality."""

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
        for _, values in param_sets.items():
            assert len(values) == num_samples

        # Check that values are within the specified ranges
        for param_name, values in param_sets.items():
            min_val, max_val = param_ranges[param_name]
            assert np.all(values >= min_val)
            assert np.all(values <= max_val)

    def test_training_data_generation(self, set_random_seeds, gaussian_function_fixture):
        """Test that training data is generated correctly."""
        # Define parameter ranges
        param_ranges = {"amplitude": (0, 10), "center": (-5, 5), "width": (0.1, 2)}

        # Define sampling points
        independent_vars_sampling = {"x": np.linspace(-10, 10, 100)}

        # Generate parameter sets
        num_samples = 10  # Small number for testing
        param_sets = mock_generate_parameter_sets(param_ranges, num_samples)

        # Generate training data
        outputs, param_values = mock_generate_training_data(
            gaussian_function_fixture, param_sets, independent_vars_sampling
        )

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
            for _, values in output.items():
                assert values is not None
                assert len(values) > 0


# Tests for the SyntheticDataGenerator class
class TestSyntheticDataGenerator:
    """Tests for the SyntheticDataGenerator class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.param_ranges = {
            "a": (1.0, 5.0),
            "b": (-3.0, 3.0),
            "c": (0.5, 2.0),
        }
        self.x_values = np.linspace(-10, 10, 100)
        self.independent_vars_sampling = {"x": self.x_values}
        self.generator = SyntheticDataGenerator(
            function=gaussian_test_func,
            param_ranges=self.param_ranges,
            independent_vars_sampling=self.independent_vars_sampling,
        )
        self.canonical_generator = SyntheticDataGenerator(
            function=gaussian_test_func,
            param_ranges=self.param_ranges,
            independent_vars_sampling=self.independent_vars_sampling,
            make_canonical=make_canonical_test,
        )

    def test_initialization(self):
        """Test initialization with valid parameters."""
        assert self.generator.function == gaussian_test_func
        assert self.generator.param_ranges == self.param_ranges
        assert self.generator.independent_vars_sampling == self.independent_vars_sampling
        assert self.generator.param_names == list(self.param_ranges.keys())
        assert self.generator.independent_var_names == list(self.independent_vars_sampling.keys())
        assert self.generator.make_canonical is None

        # Test with canonical transformation
        assert self.canonical_generator.make_canonical == make_canonical_test

    def test_initialization_errors(self):
        """Test error handling during initialization."""
        # Test with missing parameter in function signature
        invalid_params = {
            "a": (1.0, 5.0),
            "b": (-3.0, 3.0),
            "d": (0.5, 2.0),  # 'd' is not in gaussian_test_func signature
        }
        with pytest.raises(ValueError, match="Parameters .* not found in function signature"):
            SyntheticDataGenerator(
                function=gaussian_test_func,
                param_ranges=invalid_params,
                independent_vars_sampling=self.independent_vars_sampling,
            )

        # Test with missing independent variable in function signature
        invalid_vars = {"y": np.linspace(-10, 10, 100)}  # 'y' is not in gaussian_test_func signature
        with pytest.raises(ValueError, match="Independent variable .* not found in function signature"):
            SyntheticDataGenerator(
                function=gaussian_test_func,
                param_ranges=self.param_ranges,
                independent_vars_sampling=invalid_vars,
            )

    def test_generate_random_parameters(self, set_random_seeds):
        """Test random parameter generation."""
        n_samples = 100
        params = self.generator.generate_random_parameters(n_samples, canonical=False)

        # Check shape
        assert params.shape == (n_samples, len(self.param_ranges))

        # Check parameter bounds
        for i, param_name in enumerate(self.generator.param_names):
            min_val, max_val = self.param_ranges[param_name]
            assert np.all(params[:, i] >= min_val)
            assert np.all(params[:, i] <= max_val)

    def test_canonical_transformation(self, set_random_seeds):
        """Test canonical transformation of parameters."""
        # Create a generator with canonical transformation
        n_samples = 100

        # Test with canonical transformation
        params = self.canonical_generator.generate_random_parameters(n_samples, canonical=True)

        # Check that all c values are positive
        c_index = self.canonical_generator.param_names.index("c")
        assert np.all(params[:, c_index] > 0)

        # Without canonical transformation
        params = self.canonical_generator.generate_random_parameters(n_samples, canonical=False)
        # There should be some negative c values (though unlikely with our range)
        # but we can't test for their existence reliably

    def test_evaluate_function(self):
        """Test function evaluation with parameters."""
        params = np.array(
            [
                [2.0, 0.0, 1.0],  # a=2, b=0, c=1
                [3.0, 1.0, 0.5],  # a=3, b=1, c=0.5
            ]
        )

        results = self.generator.evaluate_function(params)

        # Check results structure
        assert "x" in results
        x_values, y_values = results["x"]
        assert x_values.shape == (len(self.x_values),)
        assert y_values.shape == (len(params), len(self.x_values))

        # Check specific values
        expected1 = params[0, 0] * np.exp(-((self.x_values - params[0, 1]) ** 2) / (2 * params[0, 2] ** 2))
        expected2 = params[1, 0] * np.exp(-((self.x_values - params[1, 1]) ** 2) / (2 * params[1, 2] ** 2))

        np.testing.assert_allclose(y_values[0], expected1)
        np.testing.assert_allclose(y_values[1], expected2)

    def test_generate_dataset(self, set_random_seeds):
        """Test dataset generation with and without noise."""
        n_samples = 50

        # Without noise
        params, y_data = self.generator.generate_dataset(n_samples, add_noise=False)

        assert params.shape == (n_samples, len(self.param_ranges))
        assert "x" in y_data
        x_values, y_values = y_data["x"]
        assert y_values.shape == (n_samples, len(self.x_values))

        # With noise
        params, y_data_noisy = self.generator.generate_dataset(n_samples, add_noise=True, noise_level=0.1)

        # Check that noisy data is different from clean data
        clean_params, clean_y_data = self.generator.generate_dataset(n_samples, add_noise=False)

        # Use the same parameters to generate both datasets for this test
        clean_y_values = self.generator.evaluate_function(params)["x"][1]
        noisy_x_values, noisy_y_values = y_data_noisy["x"]

        # Check that noise was added (values should be different)
        assert not np.allclose(clean_y_values, noisy_y_values)

        # Check that the difference is reasonable given the noise level
        noise_diff = np.abs(clean_y_values - noisy_y_values)
        data_range = np.max(clean_y_values) - np.min(clean_y_values)

        # Most differences should be less than 3 standard deviations of noise
        # Using 0.1 as noise_level and 3 standard deviations
        assert np.percentile(noise_diff.flatten(), 99) < 3 * 0.1 * data_range

    def test_multiple_independent_vars(self):
        """Test behavior with multiple independent variables."""
        # Set up a generator with multiple independent vars
        multi_vars = {"x": np.linspace(-5, 5, 50), "y": np.linspace(0, 10, 30)}

        generator = SyntheticDataGenerator(
            function=lambda x, y, a, b, c: a * (x + y) + b * x + c,
            param_ranges=self.param_ranges,
            independent_vars_sampling=multi_vars,
        )

        # Check that multiple independent vars raises NotImplementedError
        params = generator.generate_random_parameters(5)
        with pytest.raises(NotImplementedError, match="Multiple independent variables not yet implemented"):
            generator.evaluate_function(params)


# Integration tests ensuring compatibility between old and new implementations
class TestLegacyCompatibility:
    """Tests ensuring compatibility with legacy code."""

    def test_compare_parameter_generation(self, set_random_seeds):
        """Compare mock and SyntheticDataGenerator parameter generation."""
        param_ranges = {"a": (1.0, 5.0), "b": (-3.0, 3.0), "c": (0.5, 2.0)}
        n_samples = 50

        # Generate parameters using both methods
        mock_params = mock_generate_parameter_sets(param_ranges, n_samples)

        generator = SyntheticDataGenerator(
            function=gaussian_test_func,
            param_ranges=param_ranges,
            independent_vars_sampling={"x": np.linspace(-10, 10, 100)},
        )
        generator_params = generator.generate_random_parameters(n_samples, canonical=False)

        # Reshape generator_params into a dictionary like mock_params
        generator_params_dict = {}
        for i, param_name in enumerate(generator.param_names):
            generator_params_dict[param_name] = generator_params[:, i]

        # Verify both methods generate parameters within the same ranges
        for param_name in param_ranges:
            min_val, max_val = param_ranges[param_name]
            assert np.all(mock_params[param_name] >= min_val)
            assert np.all(mock_params[param_name] <= max_val)
            assert np.all(generator_params_dict[param_name] >= min_val)
            assert np.all(generator_params_dict[param_name] <= max_val)
