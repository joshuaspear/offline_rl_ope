import torch
from .MetricBase import MetricBase
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from ..types import WeightTensor

__all__ = [
    "WeightStd"
]

class WeightStd(MetricBase): 
            
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        weights:WeightTensor, 
        weight_msk:WeightTensor
        ) -> float:
        sum_weights = torch.mul(weights,weight_msk).sum(dim=1)
        return torch.std(sum_weights).item()