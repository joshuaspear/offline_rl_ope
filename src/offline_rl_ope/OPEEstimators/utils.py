from abc import ABCMeta, abstractmethod
from typing import Any
import torch


class WeightNorm(metaclass=ABCMeta):
    
    @abstractmethod
    def __call__(self, traj_is_weights:torch.Tensor, is_msk:torch.Tensor
                 ) -> torch.Tensor:
        pass
    
class WISNormWeights(WeightNorm):
    
    def __init__(self, smooth_eps:float=0.0, *args, **kwargs) -> None:
        self.smooth_eps = smooth_eps
    
    def calc_norm(self, traj_is_weights:torch.Tensor, is_msk:torch.Tensor
                  ) -> torch.Tensor:
        """Calculates the denominator for weighted importance sampling i.e.
        w_{t} = 1/n sum_{i=1}^{n} p_{1:t}. Note, if traj_is_weights represent
        vanilla IS samples then this will be w_{t} = 1/n sum_{i=1}^{n} p_{1:H}
        for all samples. is_msk.sum(axis=0, keepdim=True) is taken as the 
        denominator since it is required to take the average over valid time t 
        importance ratios. This may differ for different episodes.
        ref: http://proceedings.mlr.press/v48/jiang16.pdf
        smooth_eps prevents nan values occuring in instances where there exists
        valid time t importance ratios however, these are all 0. This should
        be set as small as possible. 

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
        denom:torch.Tensor = traj_is_weights.sum(axis=0, keepdim=True)
        denom = (denom+self.smooth_eps)/(
            is_msk.sum(axis=0, keepdim=True)+self.smooth_eps)
        return denom
    
    def __call__(self, traj_is_weights:torch.Tensor, is_msk:torch.Tensor
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
        denom = self.calc_norm(traj_is_weights=traj_is_weights, is_msk=is_msk)
        res = traj_is_weights/(denom+self.smooth_eps)
        return res
    
class NormWeightsPass(WeightNorm):
    
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    def __call__(self, traj_is_weights:torch.Tensor, is_msk:torch.Tensor
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
            torch.Tensor: Identical tensor to traj_is_weights
        """
        return traj_is_weights

def clip_weights(traj_is_weights:torch.Tensor, clip:float)->torch.Tensor:
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
    res = traj_is_weights.clamp(min=1/clip, max=clip)
    return res

def clip_weights_pass(traj_is_weights:torch.Tensor, clip:float)->torch.Tensor:
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
    return traj_is_weights