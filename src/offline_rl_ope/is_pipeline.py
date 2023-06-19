import logging
import numpy as np
import os
import pickle
import torch
from typing import Dict, List, Tuple

from .components.ImportanceSampling import ImportanceSampling

logger = logging.getLogger("offline_rl_ope")

def eval_weight_array(weight_res:torch.Tensor, discnt_reward_res:torch.Tensor,
                      save_dir:str=None, prefix:str=None, 
                      clip:float=None)->Tuple:
    
    losses = []
    if clip is not None:
        def _clip_is_weight(is_weight:float):
            res = min(is_weight, clip)
            res = max(res, (1/clip))
            return res
    else:
        def _clip_is_weight(is_weight:float):
            return is_weight

    clip_weight_res = torch.Tensor(list(map(_clip_is_weight, weight_res)))
    losses = weight_res*discnt_reward_res
    loss = float(torch.sum(losses))
    
    clip_losses = clip_weight_res*discnt_reward_res
    clip_loss = float(torch.sum(clip_losses))
    
    if (save_dir is not None) and (prefix is not None):
        with open(os.path.join(save_dir, "{}_losses.pkl".format(prefix)), 
                  "wb") as file:
            pickle.dump(losses, file)
        with open(os.path.join(save_dir, "{}_clip_losses.pkl".format(prefix)), 
                  "wb") as file:
            pickle.dump(clip_losses, file)
        with open(os.path.join(save_dir, "{}_weights.pkl".format(prefix)), 
                  "wb") as file:
            pickle.dump(weight_res, file)
        with open(os.path.join(save_dir, "{}_clip_weights.pkl".format(prefix)), 
                  "wb") as file:
            pickle.dump(clip_weight_res, file)

    return loss, losses, clip_loss, clip_losses, weight_res, clip_weight_res


def torch_is_evaluation(importance_sampler:ImportanceSampling, 
                        dataset:torch.utils.data.Dataset, 
                        norm_weights:bool=False, save_dir:str=None, 
                        prefix:str=None, clip:float=None)->Tuple:
    ws, rs, ncs = importance_sampler.get_dataset_w_r(dataset=dataset)
    
    logger.debug("ws: {}".format(ws))
    logger.debug("rs: {}".format(rs))
    logger.debug("ncs: {}".format(ncs))
    
    if norm_weights:
        norm_val = torch.sum(torch.Tensor(ncs))
        ws = [w/norm_val for w in ws]
    else:
        ws = [w/len(dataset) for w in ws]
    
    ws = torch.concat(ws)
    rs = torch.concat(rs)
    
    res = eval_weight_array(weight_res=ws, discnt_reward_res=rs,
                            save_dir=save_dir, prefix=prefix, 
                            clip=clip)
    return res