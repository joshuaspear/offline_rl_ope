import torch
import numpy as np
from .MetricBase import MetricBase
from ..RuntimeChecks import check_array_dim

__all__ = [
    "EffectiveSampleSize"
]

class EffectiveSampleSize(MetricBase):
    
    def __init__(self, nan_if_all_0:bool=True) -> None:
        self.__nan_if_all_0 = nan_if_all_0
        
    def __ess(self, weights:torch.Tensor) -> float:        
        # https://victorelvira.github.io/papers/kong92.pdf
        assert isinstance(weights,torch.Tensor)
        check_array_dim(weights,2)
        all_0 = (weights == 0).all().item()
        if (all_0) and (self.__nan_if_all_0):
            res = np.nan
        else:
            weights = weights.sum(dim=1)
            numer = len(weights)
            w_var = torch.var(weights).item()
            res = (numer/(1+w_var))
        return res
        
    
    def __call__(self, weights:torch.Tensor) -> float:
        return self.__ess(weights=weights)