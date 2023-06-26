import torch


def wis_norm_weights(traj_is_weights:torch.Tensor, is_msk:torch.Tensor
                     )->torch.Tensor:
    """Normalised propensity weights according to 'weighted importance sampling'

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
    denom:torch.Tensor = traj_is_weights.sum(axis=0, keepdim=True)
    res = traj_is_weights/denom
    return res


def norm_weights_pass(traj_is_weights:torch.Tensor, is_msk:torch.Tensor
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