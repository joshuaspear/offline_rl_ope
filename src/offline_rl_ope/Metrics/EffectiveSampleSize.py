import torch

from ..components.ImportanceSampler import ImportanceSampler

__all__ = [
    "EffectiveSampleSize"
]

class EffectiveSampleSize: 
    
    def __init__(self, is_obj:ImportanceSampler) -> None:
        self.__is_obj = is_obj
    
    def __ess(self) -> float:
        numer = torch.sum(torch.pow(self.__is_obj.traj_is_weights,2))
        denom = torch.pow(torch.sum(self.__is_obj.traj_is_weights),2)
        return (numer/denom).item()
        
    
    def __call__(self) -> float:
        return self.__ess()