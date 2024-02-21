import torch

from ..components.ImportanceSampler import ImportanceSampler

__all__ = [
    "EffectiveSampleSize"
]

class EffectiveSampleSize: 
    
    def __init__(self, is_obj:ImportanceSampler) -> None:
        self.__is_obj = is_obj
    
    def __ess(self) -> float:        
        # https://victorelvira.github.io/papers/kong92.pdf
        weights = self.__is_obj.traj_is_weights.sum(dim=1)
        numer = len(weights)
        return (numer/(1+torch.var(weights))).item()
        
    
    def __call__(self) -> float:
        return self.__ess()