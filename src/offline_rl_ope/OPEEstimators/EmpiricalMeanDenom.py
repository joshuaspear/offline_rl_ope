from abc import ABCMeta, abstractmethod
import torch
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker

from ..types import WeightTensor
from .utils import get_traj_weight_final


class EmpiricalMeanDenomBase(metaclass=ABCMeta):
        
    @abstractmethod
    def __call__(
        self,
        weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> Float[torch.Tensor,""]:
        pass

class EmpiricalMeanDenom(EmpiricalMeanDenomBase):
    
    def __init__(self) -> None:
        """Empirical mean denominator:
        - http://incompleteideas.net/papers/PSS-00.pdf (Q^{IS} when weights are IS)
        - http://incompleteideas.net/papers/PSS-00.pdf (Q^{PD} when weights are PD)
        """
        super().__init__()
    
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> Float[torch.Tensor,""]:
        return torch.tensor(weights.shape[0]).float()


# class WeightedEmpiricalMeanDenom(EmpiricalMeanDenomBase):
    
#     def __init__(
#         self, 
#         smooth_eps:float=0.0, 
#         cumulative:bool=False
#         ) -> None:
#         """Empirical mean denominator:
#         - http://incompleteideas.net/papers/PSS-00.pdf 
#             (Q^{ISW} when weights are IS)
#         - http://incompleteideas.net/papers/PSS-00.pdf 
#             (Q^{PDW} when weights are PD and cumulative = True)
#         """
#         super().__init__()
#         self.cumulative = cumulative
#         self.smooth_eps = smooth_eps
    
#     @jaxtyped(typechecker=typechecker)
#     def __call__(
#         self, 
#         weights:WeightTensor, 
#         is_msk:WeightTensor
#         ) -> Float[torch.Tensor,""]:
#         if self.cumulative:
#             # For each timepoint, sum across the trajectories
#             denom = torch.mul(weights,is_msk).sum()
#         else:
#             denom = get_traj_weight_final(weights=weights, is_msk=is_msk)
#             denom = denom.sum()
#         denom = denom + self.smooth_eps
#         return denom
            
