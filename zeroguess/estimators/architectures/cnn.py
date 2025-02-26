"""
CNN architecture for ZeroGuess (Future Work).

Note: This architecture is currently not implemented and is planned for future work.
"""
from typing import Dict, List, Any, Optional, Tuple

import torch
import torch.nn as nn

from zeroguess.estimators.architectures.base import BaseArchitecture


class CNNArchitecture(BaseArchitecture):
    """CNN architecture implementation (Future Work)."""
    
    def __init__(self, **params):
        """Initialize the CNN architecture with specific parameters.
        
        Args:
            **params: Architecture-specific parameters
        """
        raise NotImplementedError(
            "CNN architecture is not yet implemented. "
            "This is planned for future work."
        )
    
    def create_network(self, n_input_features: int, n_output_params: int) -> nn.Module:
        """Create a CNN network with the specified input and output dimensions.
        
        Args:
            n_input_features: Number of input features
            n_output_params: Number of output parameters
            
        Returns:
            A CNN module
        """
        raise NotImplementedError(
            "CNN architecture is not yet implemented. "
            "This is planned for future work."
        )
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Get the default parameters for the CNN architecture.
        
        Returns:
            Dictionary of default parameter values
        """
        return {
            "filters": [16, 32, 64],
            "kernel_sizes": [3, 3, 3],
            "dropout": 0.1,
            "fc_units": [128, 64],
        }
    
    @classmethod
    def get_description(cls) -> str:
        """Get a description of the CNN architecture.
        
        Returns:
            String description of the architecture
        """
        return "CNN: Convolutional Neural Network for extracting spatial features (Future Work)." 