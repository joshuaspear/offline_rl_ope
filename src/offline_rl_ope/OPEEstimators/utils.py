from abc import ABCMeta, abstractmethod
from typing import Any
import torch

from ..RuntimeChecks import check_array_dim


class WeightNorm(metaclass=ABCMeta):
    
    @abstractmethod
    def __call__(
        self, 
        traj_is_weights:torch.Tensor, 
        is_msk:torch.Tensor
        ) -> torch.Tensor:
        pass

# is_msk.sum(axis=0, keepdim=True) is taken as the 
#         denominator since it is required to take the average over valid time t 
#         importance ratios. This may differ for different episodes.
# ref: http://proceedings.mlr.press/v48/jiang16.pdf


class WISWeightNorm(WeightNorm):
    
    def __init__(
        self, 
        smooth_eps:float=0.0,
        avg_denom:bool=False,
        discount:float=1.0,
        *args, 
        **kwargs
        ) -> None:
        assert isinstance(smooth_eps,float)
        assert isinstance(avg_denom,bool)
        assert isinstance(discount,float)
        self.smooth_eps = smooth_eps
        self.avg_denom = avg_denom
        self.discount = discount
    
    def calc_norm(
        self, 
        traj_is_weights:torch.Tensor, 
        is_msk:torch.Tensor
        ) -> torch.Tensor:
        """Calculates the denominator for weighted importance sampling.
        smooth_eps prevents nan values occuring in instances where there exists
        valid time t importance ratios however, these are all 0. This should
        be set as small as possible. 
        avg_denom: defines the denominator as the average weight for time t
        as per http://proceedings.mlr.press/v48/jiang16.pdf
        
        Note:
        - If traj_is_weights represents vanilla IS samples then:
            - The denominator will be w_{t} = sum_{i=1}^{n} p_{1:H} for all 
            samples.
            - If avg_denom is set to true, the denominator will be 
            w_{t} = 1/n_{t} sum_{i=1}^{n} p_{1:H} where n_{t} is the number of 
            trajectories of at least length, t.
        - If traj_is_weights represents PD IS samples then: 
            - The denominator will be w_{t} = sum_{i=1}^{n} p_{1:t}.
            - If avg_denom is set to true, the denominator will be 
            w_{t} = 1/n_{t} sum_{i=1}^{n} p_{1:t} where n_{t} is the number of 
            trajectories of at least length, t. This definition aligns with 
            http://proceedings.mlr.press/v48/jiang16.pdf
        Args:
            traj_is_weights (torch.Tensor): (# trajectories, max(traj_length)) 
                Tensor. traj_is_weights[i,j] defines the jth timestep propensity 
                weight for the ith trajectory
            is_msk (torch.Tensor): (# trajectories, max(traj_length)) binary 
                Tensor. weight_msk[i,j] defines whether the jth timestep of the
                ith trajectory was observed

        Returns:
            torch.Tensor: Tensor of dimension (# trajectories, 1) defining the 
            normalisation value for each timestep
        """
        assert isinstance(traj_is_weights,torch.Tensor)
        assert isinstance(is_msk,torch.Tensor)
        assert traj_is_weights.shape == is_msk.shape
        check_array_dim(traj_is_weights,2)
        check_array_dim(is_msk,2)
        discnt_tens = torch.full(traj_is_weights.shape, self.discount)
        discnt_pows = torch.arange(0, traj_is_weights.shape[1])[None,:].repeat(
            traj_is_weights.shape[0],1)
        discnt_tens = torch.pow(discnt_tens,discnt_pows)
        traj_is_weights = torch.mul(traj_is_weights,discnt_tens)
        denom = (
            traj_is_weights.sum(dim=0, keepdim=True) + self.smooth_eps
            )
        if self.avg_denom:
            denom = denom/(
                is_msk.sum(dim=0, keepdim=True)+self.smooth_eps)
        return denom

    def __call__(
        self, 
        traj_is_weights:torch.Tensor, 
        is_msk:torch.Tensor
        ) -> torch.Tensor:
        """Normalised propensity weights according to 
        'weighted importance sampling'. 

        Args:
            traj_is_weights (torch.Tensor): (# trajectories, max(traj_length)) 
                Tensor. traj_is_weights[i,j] defines the jth timestep propensity 
                weight for the ith trajectory
            is_msk (torch.Tensor): (# trajectories, max(traj_length)) binary 
                Tensor. weight_msk[i,j] defines whether the jth timestep of the
                ith trajectory was observed

        Returns:
            torch.Tensor: Tensor of dimension (# trajectories, max(traj_length)) 
            with normalised weights
        """
        assert isinstance(traj_is_weights,torch.Tensor)
        assert isinstance(is_msk,torch.Tensor)
        assert traj_is_weights.shape == is_msk.shape
        check_array_dim(traj_is_weights,2)
        check_array_dim(is_msk,2)
        denom = self.calc_norm(traj_is_weights=traj_is_weights, is_msk=is_msk)
        res = traj_is_weights/denom
        return res


    
class VanillaNormWeights(WeightNorm):
    
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def __call__(
        self, 
        traj_is_weights:torch.Tensor, 
        is_msk:torch.Tensor
        )->torch.Tensor:
        """Helper function

        Args:
            traj_is_weights (torch.Tensor): (# trajectories, max(traj_length)) 
                Tensor. traj_is_weights[i,j] defines the jth timestep propensity 
                weight for the ith trajectory
            is_msk (torch.Tensor): (# trajectories, max(traj_length)) binary 
                Tensor. weight_msk[i,j] defines whether the jth timestep of the
                ith trajectory was observed

        Returns:
            torch.Tensor: traj_is_weights with element wise average
        """
        assert isinstance(traj_is_weights,torch.Tensor)
        check_array_dim(traj_is_weights,2)
        # The first dimension defines the number of trajectories and we require
        # the average over trajectories
        return traj_is_weights/traj_is_weights.shape[0]

def clip_weights(
    traj_is_weights:torch.Tensor, 
    clip:float
    )->torch.Tensor:
    """Clips propensity weights according to the value provided in clip

    Args:
        traj_is_weights (torch.Tensor): (# trajectories, max(traj_length)) 
            Tensor. traj_is_weights[i,j] defines the jth timestep propensity 
            weight for the ith trajectory
        clip (float): [clip,1/clip] defines the max and min values the 
            propensity weights may take

    Returns:
        torch.Tensor: Tensor of dimension (# trajectories, max(traj_length)) 
            with clipped weights
    """
    assert isinstance(traj_is_weights,torch.Tensor)
    assert isinstance(clip,float)
    res = traj_is_weights.clamp(min=1/clip, max=clip)
    return res

def clip_weights_pass(
    traj_is_weights:torch.Tensor, 
    clip:float
    )->torch.Tensor:
    """Helper function

    Args:
        traj_is_weights (torch.Tensor): (# trajectories, max(traj_length)) 
            Tensor. traj_is_weights[i,j] defines the jth timestep propensity 
            weight for the ith trajectory
        clip (float): [clip,1/clip] defines the max and min values the 
            propensity weights may take

    Returns:
        torch.Tensor: Identical tensor to traj_is_weights
    """
    assert isinstance(traj_is_weights,torch.Tensor)
    return traj_is_weights