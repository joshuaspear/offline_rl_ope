import numpy as np
import os
import pickle
import torch
from typing import Dict, List, Tuple

from .components.ImportanceSampling import ImportanceSampling


def get_weight_array(importance_sampler:ImportanceSampling, 
                     dataset:List[Dict[str,np.array]]):
    weight_res = []
    discnt_reward_res = []
    for i, vals in enumerate(dataset):
        weight, discnt_reward = importance_sampler.get_traj_loss(
            state=torch.Tensor(vals["state"]), 
            action=torch.Tensor(vals["act"]), 
            reward=torch.Tensor(vals["reward"]))
        weight_res.append(weight)
        discnt_reward_res.append(discnt_reward)
    weight_res = np.array(weight_res)
    discnt_reward_res = np.array(discnt_reward_res)
    return weight_res, discnt_reward_res

def eval_weight_array(weight_res:np.array, discnt_reward_res:np.array, 
                      dataset_len:int, norm_weights:bool=False,
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

    clip_weight_res = np.array(list(map(_clip_is_weight, weight_res)))
    if norm_weights:
        weight_res = weight_res/weight_res.sum()
        clip_weight_res = clip_weight_res/clip_weight_res.sum()
    else:
        weight_res = weight_res/dataset_len
        clip_weight_res = clip_weight_res/dataset_len
    losses = weight_res*discnt_reward_res
    losses = torch.Tensor(losses)
    loss = float(torch.sum(losses))
    
    clip_losses = clip_weight_res*discnt_reward_res
    clip_losses = torch.Tensor(clip_losses)
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
    weight_res, discnt_reward_res = get_weight_array(
        importance_sampler=importance_sampler, dataset=dataset)
    res = eval_weight_array(weight_res=weight_res, 
                              discnt_reward_res=discnt_reward_res, 
                              dataset_len=len(dataset), 
                              norm_weights=norm_weights, 
                              save_dir=save_dir, prefix=prefix, 
                              clip=clip)
    return res