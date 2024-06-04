from abc import ABCMeta, abstractmethod
import torch
from typing import Union
import numpy as np

from ..types import TorchPolicyReturn, NumpyPolicyReturn

__all__ = [
    "PropensityTrainer"
]

class PropensityTrainer(metaclass=ABCMeta):
            
    @abstractmethod
    def save(self, path:str):
        pass
    
    @abstractmethod
    def predict_proba(
        self, 
        x: Union[torch.Tensor, np.ndarray], 
        y: Union[torch.Tensor, np.ndarray], 
        *args, 
        **kwargs
        ) -> Union[torch.Tensor, np.ndarray]:
        pass
    
    @abstractmethod
    def predict(
        self, 
        x:Union[torch.Tensor, np.ndarray],
        *args, 
        **kwargs
        ) -> Union[torch.Tensor, np.ndarray]:
        pass
    
    @abstractmethod
    def policy_func(
        self, 
        x: Union[torch.Tensor, np.ndarray], 
        y: Union[torch.Tensor, np.ndarray], 
        *args, 
        **kwargs
        ) -> Union[TorchPolicyReturn, NumpyPolicyReturn]:
        pass