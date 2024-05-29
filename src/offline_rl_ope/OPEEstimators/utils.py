from abc import ABCMeta, abstractmethod
import torch
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker

from ..types import WeightTensor


class WeightNorm(metaclass=ABCMeta):
    
    @abstractmethod
    def __call__(
        self, 
        traj_is_weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> WeightTensor:
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
        cumulative:bool=False,
        *args, 
        **kwargs
        ) -> None:
        assert isinstance(smooth_eps,float)
        assert isinstance(avg_denom,bool)
        assert isinstance(cumulative,bool)
        self.smooth_eps = smooth_eps
        self.avg_denom = avg_denom
        self.cumulative = cumulative
        
    @jaxtyped(typechecker=typechecker)    
    def calc_norm(
        self, 
        traj_is_weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> Float[torch.Tensor, "1 traj_length"]:
        """Calculates the denominator for weighted importance sampling.
        smooth_eps prevents nan values occuring in instances where there exists
        valid time t importance ratios however, these are all 0. This should
        be set as small as possible. 
        avg_denom defines the denominator as the average importance weight 
        rather than the sum of importance weights i.e.: 
            - http://proceedings.mlr.press/v48/jiang16.pdf and;
            - https://arxiv.org/pdf/2005.01643
        
        Note:
        vanilla IS samples => traj_is_weights has entries:
            $w_{i,H} = \prod_{t=0}^{H_{i}}w_{i,t}$
        - If traj_is_weights represents vanilla IS samples:
            - The denominator will be:
                $sum_{i=1}^{n} w_{i,H}$ for all samples.
            - If cumulative is True, the denominator will be:
                $sum_{i=1}^{n} w_{i,H}$ for all samples i.e., there is no 
                difference
                as the cumulative sum of weights are all the same
            - If avg_denom is set to true, the denominator will be 
                $\frac{1}{n}sum_{i=1}^{n} w_{i,H}$
        
        PD samples => traj_is_weights has entries:
            $w_{i,t'} = \prod_{t=0}^{t'}w_{i,t'}$
        - If traj_is_weights represents PD IS samples then: 
            - The denominator will be:
                $sum_{i=1}^{n} w_{i,H}$ for all samples i.e., the same as for
                vanilla IS
            - If avg_denom is set to true, the denominator will be 
                $\frac{1}{n}sum_{i=1}^{n} w_{i,H}$
            - If cumulative is True, the denominator will be:
                [i,t] entry of the weights will be $sum_{i=1}^{n} w_{i,t'}$
                i.e., the value will be the same across all trajectories,
                for a time point 
            - If avg_denom is set to true, the denominator will be 
                [i,t] entry of the weights will be 
                $\frac{1}{n}sum_{i=1}^{n} w_{i,t'}$
            
        Args:
            traj_is_weights (torch.Tensor): (# trajectories, max(traj_length)) 
                Tensor. traj_is_weights[i,j] defines the jth timestep propensity 
                weight for the ith trajectory
            is_msk (torch.Tensor): (# trajectories, max(traj_length)) binary 
                Tensor. weight_msk[i,j] defines whether the jth timestep of the
                ith trajectory was observed

        Returns:
            torch.Tensor: Tensor of dimension (1 max(traj_length)) 
            defining the normalisation value for each timestep
        """
        # assert isinstance(traj_is_weights,torch.Tensor)
        # assert isinstance(is_msk,torch.Tensor)
        # assert traj_is_weights.shape == is_msk.shape
        # check_array_dim(traj_is_weights,2)
        # check_array_dim(is_msk,2)
        if self.cumulative:
            # For each timepoint, sum across the trajectories
            denom = (
                traj_is_weights.sum(dim=0, keepdim=True) + self.smooth_eps
                )
        else:
            # Find the index of the final step for each trajectory
            _final_idx = is_msk.cumsum(dim=1).argmax(dim=1)
            # Find the associated weight of each trajectory and sum
            denom = traj_is_weights[
                torch.arange(traj_is_weights.shape[0]), _final_idx].sum()
            denom = denom.repeat((1,traj_is_weights.shape[1])) + self.smooth_eps

        if self.avg_denom:
            denom = denom/traj_is_weights.shape[0]
        return denom

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        traj_is_weights:WeightTensor, 
        is_msk:WeightTensor
        ) -> WeightTensor:
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
        # assert isinstance(traj_is_weights,torch.Tensor)
        # assert isinstance(is_msk,torch.Tensor)
        # assert traj_is_weights.shape == is_msk.shape
        # check_array_dim(traj_is_weights,2)
        # check_array_dim(is_msk,2)
        denom = self.calc_norm(traj_is_weights=traj_is_weights, is_msk=is_msk)
        res = traj_is_weights/denom
        return res


    
class VanillaNormWeights(WeightNorm):
    
    def __init__(self, *args, **kwargs) -> None:
        pass

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        traj_is_weights:WeightTensor, 
        is_msk:WeightTensor
        )->WeightTensor:
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
        # assert isinstance(traj_is_weights,torch.Tensor)
        # check_array_dim(traj_is_weights,2)
        # The first dimension defines the number of trajectories and we require
        # the average over trajectories
        return traj_is_weights/traj_is_weights.shape[0]

@jaxtyped(typechecker=typechecker)
def clip_weights(
    traj_is_weights:WeightTensor, 
    clip:float
    )->WeightTensor:
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
    # assert isinstance(traj_is_weights,torch.Tensor)
    assert isinstance(clip,float)
    res = traj_is_weights.clamp(min=1/clip, max=clip)
    return res

@jaxtyped(typechecker=typechecker)
def clip_weights_pass(
    traj_is_weights:WeightTensor, 
    clip:float
    )->WeightTensor:
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
    # assert isinstance(traj_is_weights,torch.Tensor)
    return traj_is_weights