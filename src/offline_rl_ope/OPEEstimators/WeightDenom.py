from abc import ABCMeta, abstractmethod
import torch
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from ..types import WeightTensor

class WeightDenomBase(metaclass=ABCMeta):
    
    @abstractmethod
    def __call__(
        self, 
        weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> WeightTensor:
        pass


class PassWeightDenom(WeightDenomBase):
    
    def __call__(
        self, 
        weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> WeightTensor:
        return weights


class AvgWeightDenom(WeightDenomBase):
    
    def __init__(self) -> None:
        """Weight denominator as per:
        - https://arxiv.org/pdf/1604.00923 (DR when weights are IS)
        """
        super().__init__()
    
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> WeightTensor:
        return weights/torch.tensor(weights.shape[0])
        

class PiTWeightDenom(WeightDenomBase):
    
    def __init__(
        self, 
        smooth_eps:float=0.0
        ) -> None:
        """Weight denominator as per:
        - https://arxiv.org/pdf/1906.03735 (snsis when weights are PD)
        - https://arxiv.org/pdf/1604.00923 (WDR when weights are PD)

        Args:
            smooth_eps (float, optional): Laplacian smoothing. Defaults to 0.0.
        """
        super().__init__()
        assert isinstance(smooth_eps,float)
        self.smooth_eps = smooth_eps

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> WeightTensor:
        msked_weights = torch.mul(weights, is_msk)
        pit_vals = msked_weights.mean(dim=0,keepdim=True).repeat(
            weights.shape[0], 1
        )
        pit_vals = pit_vals + self.smooth_eps
        res = msked_weights/pit_vals
        return res

