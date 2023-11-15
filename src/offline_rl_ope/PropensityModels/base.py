from abc import ABCMeta, abstractmethod
import numpy as np
from ..types import PropensityEstimatorType, Float32NDArray

__all__ = [
    "PropensityTrainer"
]

class PropensityTrainer(metaclass=ABCMeta):
    
    def __init__(self, estimator:PropensityEstimatorType) -> None:
        self.estimator = estimator
    
    @abstractmethod
    def predict(
        self, 
        x:np.array, 
        *args, 
        **kwargs
        ) -> Float32NDArray:
        """Outputs the y values with highest likelihood given x

        Args:
            x (np.array): Array of dimension ...

        Returns:
            np.array: Array of dimension ...
        """
        pass
    
    @abstractmethod
    def predict_proba(
        self, 
        x:Float32NDArray, 
        y:Float32NDArray, 
        *args,
        **kwargs
        )->Float32NDArray:
        """Outputs the un-normalised/normalised likelihood of each dimension of
        y given input x for regression/classification, respectively.

        Args:
            x (np.array): Array of dimension ...
            y (np.array): Array of dimension ...

        Returns:
            np.array: Array of dimension ...
        """
        pass
    
    @abstractmethod
    def save(self, path:str):
        pass