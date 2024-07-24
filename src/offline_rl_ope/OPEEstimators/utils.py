from abc import ABCMeta, abstractmethod
import torch
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker

from ..types import WeightTensor


def get_traj_weight_final(
    weights:WeightTensor,
    is_msk:WeightTensor
    )->Float[torch.Tensor, "n_trajectories"]:
    # Find the index of the final step for each trajectory
    _final_idx = is_msk.cumsum(dim=1).argmax(dim=1)
    # Find the associated weight of each trajectory and sum
    return weights[torch.arange(weights.shape[0]), _final_idx]


@jaxtyped(typechecker=typechecker)
def clip_weights(
    weights:WeightTensor, 
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
    res = weights.clamp(min=1/clip, max=clip)
    return res

@jaxtyped(typechecker=typechecker)
def clip_weights_pass(
    weights:WeightTensor, 
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
    return weights