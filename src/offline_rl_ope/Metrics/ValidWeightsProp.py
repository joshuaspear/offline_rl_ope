import torch

from ..components.ImportanceSampler import ImportanceSampler

__all__ = [
    "ValidWeightsProp"
]

class ValidWeightsProp: 
    
    def __init__(
        self, 
        is_obj:ImportanceSampler, 
        min_w:float, 
        max_w:float
        ) -> None:
        self.__is_obj = is_obj
        self.__min_w = min_w
        self.__max_w = max_w
    
    def __valid_weights(self) -> float:
        vw_mask = (
            (self.__is_obj.traj_is_weights > self.__min_w) & 
            (self.__is_obj.traj_is_weights < self.__max_w)
            )
        vw_num = torch.sum(vw_mask, axis=1)
        vw_denom = torch.sum(self.__is_obj.is_weight_calc.weight_msk, axis=1)
        return torch.mean(vw_num/vw_denom).item()
        
    def __call__(self) -> float:
        return self.__valid_weights()