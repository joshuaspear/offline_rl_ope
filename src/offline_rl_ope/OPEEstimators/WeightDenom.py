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


class PiTWeightDenomBase(WeightDenomBase):
    
    def __init__(
        self, 
        smooth_eps:float=0.0
        ) -> None:
        """
        The logic follows as:
        - torch.mul(weights, is_msk) masks missing values;
        - msked_weights.mean( defines the mean value across trajectories for 
        each time step. The repeat projects the same value across each trajectory
        for the same timestep.
        - msked_weights/pit_vals defines the resulting weights
        
        When the input is vanilla, weights will contain the same weight value 
        for each time-step and thus the output will be identical to not using 
        point in time.
        
        Args:
            smooth_eps (float, optional): Laplacian smoothing. Defaults to 0.0.
        """
        super().__init__()
        assert isinstance(smooth_eps,float)
        self.smooth_eps = smooth_eps
    
    @jaxtyped(typechecker=typechecker)
    def get_pit_denom(
        self, 
        msked_weights:WeightTensor,
        ) -> WeightTensor:
        pit_vals = msked_weights.sum(dim=0,keepdim=True).repeat(
            msked_weights.shape[0], 1
        )
        pit_vals = pit_vals + self.smooth_eps
        return pit_vals

    @abstractmethod
    def __call__(
        self, 
        weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> WeightTensor:
        pass

class AvgPiTWeightDenom(PiTWeightDenomBase):
    
    def __init__(self, smooth_eps = 0.0):
        super().__init__(smooth_eps)
    
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> WeightTensor:
        msked_weights = torch.mul(weights, is_msk)
        pit_denom = self.get_pit_denom(msked_weights=msked_weights)
        pit_denom = pit_denom/weights.shape[0]
        res = msked_weights/pit_denom
        return res
    

class SumPiTWeightDenom(PiTWeightDenomBase):
    
    def __init__(self, smooth_eps = 0.0):
        super().__init__(smooth_eps)

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> WeightTensor:
        msked_weights = torch.mul(weights, is_msk)
        pit_denom = self.get_pit_denom(msked_weights=msked_weights)
        res = msked_weights/pit_denom
        return res