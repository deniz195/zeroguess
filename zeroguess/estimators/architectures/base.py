"""
Base interface for neural network architectures.
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional

import torch
import torch.nn as nn


class BaseArchitecture(ABC):
    """Base class for all neural network architectures in ZeroGuess.
    
    This abstract class defines the interface that all neural network architectures
    must implement. It provides methods for creating and configuring networks.
    """
    
    @abstractmethod
    def create_network(self, n_input_features: int, n_output_params: int) -> nn.Module:
        """Create a neural network with the specified input and output dimensions.
        
        Args:
            n_input_features: Number of input features
            n_output_params: Number of output parameters
            
        Returns:
            A PyTorch neural network module
        """
        pass
    
    @classmethod
    def get_default_params(cls) -> Dict[str, Any]:
        """Get the default parameters for this architecture.
        
        Returns:
            Dictionary of default parameter values
        """
        return {}
    
    @classmethod
    def validate_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        """Validate architecture-specific parameters and apply defaults.
        
        Args:
            params: Dictionary of architecture-specific parameters
            
        Returns:
            Dictionary of validated parameters with defaults applied
            
        Raises:
            ValueError: If any parameter is invalid
        """
        # Start with default parameters
        validated = cls.get_default_params()
        
        # Override with provided parameters
        for key, value in params.items():
            if key in validated:
                validated[key] = value
            else:
                raise ValueError(f"Unknown parameter '{key}' for {cls.__name__}")
        
        return validated
    
    @classmethod
    def get_name(cls) -> str:
        """Get the name of this architecture.
        
        Returns:
            String name of the architecture
        """
        # Default implementation uses the class name without the "Architecture" suffix
        name = cls.__name__
        if name.endswith("Architecture"):
            name = name[:-12]  # Remove "Architecture" suffix
        return name.lower()
    
    @classmethod
    def get_description(cls) -> str:
        """Get a description of this architecture.
        
        Returns:
            String description of the architecture
        """
        return "Base neural network architecture" 