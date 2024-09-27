import torch
from .MetricBase import MetricBase
from ..OPEEstimators.utils import get_traj_weight_final
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
        fnl_weights = get_traj_weight_final(
            weights=weights,
            is_msk=weight_msk
            )
        vw_mask = (
            (fnl_weights > self.__min_w) & 
            (fnl_weights < self.__max_w)
            ).squeeze()
        return torch.mean(vw_mask.float()).item()
        
    def __call__(
        self, 
        weights:WeightTensor, 
        weight_msk:WeightTensor
        ) -> float:
        return self.__valid_weights(
            weights=weights, 
            weight_msk=weight_msk
            )