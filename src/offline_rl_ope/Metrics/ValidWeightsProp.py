import torch
from .MetricBase import MetricBase
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from ..types import WeightTensor

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
    
    @jaxtyped(typechecker=typechecker)
    def __valid_weights(
        self, 
        weights:WeightTensor, 
        weight_msk:WeightTensor
        ) -> float:
        # assert isinstance(weights,torch.Tensor)
        # assert isinstance(weight_msk,torch.Tensor)
        # check_array_dim(weights,2)
        # check_array_dim(weight_msk,2)
        # assert weights.shape == weight_msk.shape
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
        weights:WeightTensor, 
        weight_msk:WeightTensor
        ) -> float:
        return self.__valid_weights(
            weights=weights, 
            weight_msk=weight_msk
            )