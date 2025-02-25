"""
Integration with lmfit's Model class.
"""
from typing import Any, Callable, Dict, Optional, Tuple, Union, List
import inspect
import numpy as np

try:
    import lmfit
except ImportError:
    raise ImportError(
        "lmfit is required for this functionality. "
        "Please install it with `pip install lmfit`."
    )

import zeroguess


class Model(lmfit.Model):
    """Enhanced version of lmfit.Model with automatic parameter estimation.
    
    This class extends lmfit's Model class to provide automatic initial parameter
    estimation using ZeroGuess. It trains a parameter estimator when the model is
    created and uses it to provide initial parameter values when fitting.
    
    Example:
        ```python
        import numpy as np
        from zeroguess.integration import lmfit_integration
        
        def gaussian(x, amplitude, center, width):
            return amplitude * np.exp(-(x - center)**2 / (2 * width**2))
        
        # Define sampling points for training
        x_sampling = np.linspace(-10, 10, 100)
        
        # Create model with automatic parameter estimation
        model = lmfit_integration.Model(
            gaussian,
            param_ranges={
                'amplitude': (0, 10),
                'center': (-5, 5),
                'width': (0.1, 2)
            },
            independent_vars_sampling={
                'x': x_sampling
            }
        )
        
        # Fit data with automatic parameter estimation
        result = model.fit(y_data, x=x_data)
        ```
    """
    
    def __init__(
        self,
        func: Callable,
        param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
        independent_vars_sampling: Optional[Dict[str, np.ndarray]] = None,
        independent_vars: Optional[List[str]] = None,
        prefix: str = '',
        name: Optional[str] = None,
        **kwargs
    ):
        """Initialize the enhanced Model with parameter estimation capability.
        
        Args:
            func: The model function to be wrapped
            param_ranges: Dictionary mapping parameter names to (min, max) tuples
            independent_vars_sampling: Dictionary mapping independent variable names
                to arrays of sampling points for training
            independent_vars: Names of independent variables (passed to lmfit.Model)
            prefix: Prefix for parameter names (passed to lmfit.Model)
            name: Name for the model (passed to lmfit.Model)
            **kwargs: Additional keyword arguments passed to lmfit.Model
        """
        # Initialize the parent lmfit.Model
        super().__init__(func, independent_vars=independent_vars, prefix=prefix, name=name, **kwargs)
        
        # Store ZeroGuess-specific parameters
        self.param_ranges = param_ranges
        self.independent_vars_sampling = independent_vars_sampling
        self._estimator = None
        
        # If parameter ranges are provided, initialize and train the estimator
        if param_ranges is not None and independent_vars_sampling is not None:
            self._initialize_estimator()
    
    def _initialize_estimator(self):
        """Initialize and train the parameter estimator."""
        # Create estimator
        self._estimator = zeroguess.create_estimator(
            function=self.func,
            param_ranges=self.param_ranges,
            independent_vars_sampling=self.independent_vars_sampling,
        )
        
        # Train the estimator
        self._estimator.train()
    
    def fit(
        self,
        data: np.ndarray,
        params: Optional[lmfit.Parameters] = None,
        weights: Optional[np.ndarray] = None,
        method: Optional[str] = 'leastsq',
        **kwargs
    ) -> lmfit.model.ModelResult:
        """Fit the model to data with automatic parameter estimation.
        
        This method extends lmfit.Model.fit by automatically providing initial
        parameter values using ZeroGuess's parameter estimation if params is not provided.
        
        Args:
            data: The data to fit
            params: Parameters for the fit (optional, will be estimated if not provided)
            weights: Weights for the fit (optional)
            method: Fitting method (defaults to 'leastsq')
            **kwargs: Additional keyword arguments for the fit
                      including independent variables
        
        Returns:
            ModelResult from the fit
        """
        # Check if parameters were provided
        if params is None and self._estimator is not None and self.param_ranges is not None:
            # Extract independent variables from kwargs
            if len(self.independent_vars) != 1:
                raise NotImplementedError(
                    "Currently only supports models with one independent variable"
                )
            
            # Get the independent variable (typically 'x')
            indep_var_name = self.independent_vars[0]
            if indep_var_name not in kwargs:
                raise ValueError(f"Independent variable '{indep_var_name}' must be provided")
            
            x_data = kwargs[indep_var_name]
            
            # Use the estimator to predict initial parameters
            estimated_params = self._estimator.predict(x_data, data)
            
            # Create lmfit.Parameters object with estimated values
            params = self.make_params()
            for name, value in estimated_params.items():
                params[name].set(value=value)
        
        # Ensure method is not None
        if method is None:
            method = 'leastsq'
            
        # Call the parent fit method with estimated or provided parameters
        return super().fit(data, params, weights, method, **kwargs)


# Create alias for easier imports
__all__ = ["Model"]
