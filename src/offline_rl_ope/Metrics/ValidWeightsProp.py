import torch
from .MetricBase import MetricBase

__all__ = [
    "ValidWeightsProp"
]

class ValidWeightsProp(MetricBase): 
    
    def __init__(
        self, 
        min_w:float, 
        max_w:float
        ) -> None:
        self.__min_w = min_w
        self.__max_w = max_w
    
    def __valid_weights(
        self, 
        weights:torch.Tensor, 
        weight_msk:torch.Tensor
        ) -> float:
        vw_mask = (
            (weights > self.__min_w) & 
            (weights < self.__max_w)
            )
        vw_num = torch.sum(input=vw_mask, dim=1)
        vw_denom = torch.sum(
            input=weight_msk, dim=1
            )
        return torch.mean(vw_num/vw_denom).item()
        
    def __call__(
        self, 
        weights:torch.Tensor, 
        weight_msk:torch.Tensor
        ) -> float:
        return self.__valid_weights(
            weights=weights, 
            weight_msk=weight_msk
            )