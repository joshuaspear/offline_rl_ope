import torch


    
def wis_norm_weights(traj_is_weights:torch.Tensor, is_msk:torch.Tensor
                     )->torch.Tensor:
    denom_num:torch.Tensor = traj_is_weights.sum(axis=0, keepdim=True)
    dneom_denom:torch.Tensor = is_msk.sum(axis=0, keepdim=True)
    denom:torch.Tensor = denom_num/dneom_denom
    denom = denom.expand(traj_is_weights.shape)
    res = traj_is_weights/denom
    return res

def norm_weights_pass(traj_is_weights:torch.Tensor, is_msk:torch.Tensor
                      )->torch.Tensor:
    denom:torch.Tensor = is_msk.sum(axis=0, keepdim=True)
    denom = denom.expand(traj_is_weights.shape)
    res = traj_is_weights/denom
    return res

def clip_weights(traj_is_weights:torch.Tensor, clip:float):
    res = traj_is_weights.clamp(min=1/clip, max=clip)
    return res

def clip_weights_pass(traj_is_weights:torch.Tensor, clip:float):
    return traj_is_weights