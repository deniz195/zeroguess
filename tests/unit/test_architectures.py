"""
Unit tests for the neural network architecture selection feature.
"""

import unittest

import numpy as np
import torch

from zeroguess import create_estimator
from zeroguess.estimators.architectures import (
    get_architecture,
    get_architecture_info,
    list_architectures,
)
from zeroguess.estimators.architectures.cnn import CNNArchitecture
from zeroguess.estimators.architectures.mlp import MLPArchitecture


class TestArchitectureSelection(unittest.TestCase):
    """Tests for the architecture selection feature."""

    def setUp(self):
        """Set up common test data."""

        # Define a simple function for testing
        def linear_function(x, a, b):
            return a * x + b

        self.function = linear_function
        self.param_ranges = {"a": (0.5, 5.0), "b": (-2.0, 2.0)}
        self.independent_vars_sampling = {"x": np.linspace(-10, 10, 50)}

    def test_list_architectures(self):
        """Test listing available architectures."""
        architectures = list_architectures()
        self.assertIsInstance(architectures, list)
        self.assertIn("mlp", architectures)
        self.assertIn("cnn", architectures)

    def test_get_architecture_info(self):
        """Test getting architecture information."""
        info = get_architecture_info()
        self.assertIsInstance(info, dict)
        self.assertIn("mlp", info)
        self.assertIn("cnn", info)
        self.assertIn("description", info["mlp"])
        self.assertIn("description", info["cnn"])
        self.assertIn("default_params", info["mlp"])
        self.assertIn("default_params", info["cnn"])

    def test_get_architecture(self):
        """Test getting architecture instances."""
        # Get default architecture
        arch = get_architecture("best")
        self.assertIsInstance(arch, MLPArchitecture)

        # Get explicit MLP architecture
        arch = get_architecture("mlp")
        self.assertIsInstance(arch, MLPArchitecture)

        # Get with custom parameters
        arch = get_architecture("mlp", hidden_layers=[64, 32], dropout_rate=0.2)
        self.assertIsInstance(arch, MLPArchitecture)
        self.assertEqual(arch.params["hidden_layers"], [64, 32])
        self.assertEqual(arch.params["dropout_rate"], 0.2)

        # Get CNN architecture
        arch = get_architecture("cnn")
        self.assertIsInstance(arch, CNNArchitecture)

        # Get CNN with custom parameters
        arch = get_architecture("cnn", n_conv_layers=2, filters=[32, 64], kernel_size=3)
        self.assertIsInstance(arch, CNNArchitecture)
        self.assertEqual(arch.params["n_conv_layers"], 2)
        self.assertEqual(arch.params["filters"], [32, 64])
        self.assertEqual(arch.params["kernel_size"], 3)

        # Test invalid architecture
        with self.assertRaises(ValueError):
            get_architecture("invalid_architecture")

    def test_architecture_parameter_validation(self):
        """Test that all architectures properly validate parameters and reject unknown ones."""
        # Get the list of available architectures
        architectures = list_architectures()

        # Test each registered architecture
        for arch_name in architectures:
            # Try to create an architecture with an invalid parameter
            with self.assertRaises(
                ValueError,
                msg=f"{arch_name} architecture should reject unknown parameters",
            ):
                get_architecture(arch_name, invalid_parameter_name=123)

            # Try a different invalid parameter
            with self.assertRaises(
                ValueError,
                msg=f"{arch_name} architecture should reject unknown parameters",
            ):
                get_architecture(arch_name, unknown_setting="value")

            # Make sure it works with valid parameters
            try:
                # Get default parameters for this architecture
                defaults = get_architecture_info()[arch_name]["default_params"]

                if defaults:
                    # Take the first parameter and use its value
                    param_name = next(iter(defaults.keys()))
                    param_value = defaults[param_name]

                    # Should not raise error with valid parameter
                    valid_params = {param_name: param_value}
                    get_architecture(arch_name, **valid_params)
            except Exception as e:
                self.fail(
                    f"Creating {arch_name} with valid parameters raised {type(e).__name__}: {e}"
                )

    def test_create_network(self):
        """Test creating networks with different architectures."""
        # Create MLP architecture and network
        arch = get_architecture("mlp")
        network = arch.create_network(n_input_features=10, n_output_params=2)

        # Test forward pass
        x = torch.randn(5, 10)  # 5 samples, 10 features
        output = network(x)
        self.assertEqual(output.shape, (5, 2))  # 5 samples, 2 output parameters

        # Create CNN architecture and network
        arch = get_architecture("cnn")
        network = arch.create_network(n_input_features=10, n_output_params=2)

        # Test forward pass
        x = torch.randn(5, 10)  # 5 samples, 10 features
        output = network(x)
        self.assertEqual(output.shape, (5, 2))  # 5 samples, 2 output parameters

    def test_estimator_with_architecture(self):
        """Test creating an estimator with a specified architecture."""
        # Create estimator with MLP architecture
        estimator = create_estimator(
            function=self.function,
            param_ranges=self.param_ranges,
            independent_vars_sampling=self.independent_vars_sampling,
            architecture="mlp",
            architecture_params={"hidden_layers": [32, 64, 32]},
        )

        # Check that the architecture is correctly specified
        self.assertEqual(estimator.architecture_name, "mlp")
        self.assertEqual(estimator.architecture_params["hidden_layers"], [32, 64, 32])

        # Create estimator with CNN architecture
        estimator = create_estimator(
            function=self.function,
            param_ranges=self.param_ranges,
            independent_vars_sampling=self.independent_vars_sampling,
            architecture="cnn",
            architecture_params={"n_conv_layers": 2, "filters": [16, 32]},
        )

        # Check that the architecture is correctly specified
        self.assertEqual(estimator.architecture_name, "cnn")
        self.assertEqual(estimator.architecture_params["n_conv_layers"], 2)
        self.assertEqual(estimator.architecture_params["filters"], [16, 32])

        # Test with 'best' architecture - should default to MLP internally
        estimator = create_estimator(
            function=self.function,
            param_ranges=self.param_ranges,
            independent_vars_sampling=self.independent_vars_sampling,
            architecture="best",
        )

        # Check that it keeps the 'best' name but uses MLP internally
        self.assertEqual(estimator.architecture_name, "best")

    def test_input_size_validation(self):
        """Test that architectures enforce input size validation principles."""
        # Test MLP architecture input size validation
        mlp_arch = get_architecture("mlp")
        mlp_network = mlp_arch.create_network(n_input_features=10, n_output_params=2)

        # Same size should be valid
        valid_size = 10
        try:
            mlp_arch.validate_input_size(mlp_network, valid_size, valid_size)
        except ValueError:
            self.fail(
                "validate_input_size raised ValueError for valid input size in MLP"
            )

        # Different size should raise ValueError for MLP
        invalid_size = 15
        with self.assertRaises(
            ValueError, msg="MLP should reject mismatched input sizes"
        ):
            mlp_arch.validate_input_size(mlp_network, invalid_size, valid_size)

        # Test CNN architecture input size validation
        cnn_arch = get_architecture("cnn")
        cnn_network = cnn_arch.create_network(n_input_features=10, n_output_params=2)

        # Same size should be valid for CNN too
        try:
            cnn_arch.validate_input_size(cnn_network, valid_size, valid_size)
        except ValueError:
            self.fail(
                "validate_input_size raised ValueError for valid input size in CNN"
            )

        # Different size should now raise ValueError for CNN as well, after our update
        with self.assertRaises(
            ValueError, msg="CNN should reject mismatched input sizes"
        ):
            cnn_arch.validate_input_size(cnn_network, invalid_size, valid_size)


if __name__ == "__main__":
    unittest.main()
