from abc import ABCMeta, abstractmethod
from typing import Any
import torch
import numpy as np

__all__ = [
    "MetricBase"
]

class MetricBase(metaclass=ABCMeta):
        
    @abstractmethod
    def __call__(self, weights:torch.Tensor, *args:Any, **kwargs:Any) -> float:
        return self.__ess(weights=weights)