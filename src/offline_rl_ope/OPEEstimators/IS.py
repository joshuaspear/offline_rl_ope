import logging
import torch
from typing import Any, Dict, List

from .utils import (WISNormWeights, NormWeightsPass, clip_weights_pass, 
                    clip_weights)
from .base import OPEEstimatorBase

logger = logging.getLogger("offline_rl_ope")

class ISEstimatorBase(OPEEstimatorBase):
    
    def __init__(self, norm_weights:bool, clip:float=None, 
                 cache_traj_rewards:bool=False, norm_kwargs:Dict[str,Any] = {}
                 ) -> None:
        super().__init__(cache_traj_rewards)
        if norm_weights:    
            self.norm_weights = WISNormWeights(**norm_kwargs)
        else:
            self.norm_weights = NormWeightsPass(**norm_kwargs)
        self.clip = clip
        if clip:
            self.clip_weights = clip_weights
        else:
            self.clip_weights = clip_weights_pass
    
    def process_weights(self, weights:torch.Tensor, is_msk:torch.Tensor):
        assert weights.shape == is_msk.shape
        weights = self.clip_weights(
            traj_is_weights=weights, clip=self.clip)
        weights = self.norm_weights(traj_is_weights=weights, is_msk=is_msk)
        return weights
        

class ISEstimator(ISEstimatorBase):
    
    def __init__(self, norm_weights: bool, clip: float = None,
                 cache_traj_rewards:bool=False, norm_kwargs:Dict[str,Any] = {}
                 ) -> None:
        super().__init__(norm_weights=norm_weights, clip=clip, 
                         cache_traj_rewards=cache_traj_rewards, 
                         norm_kwargs=norm_kwargs)
        
    def get_dataset_discnt_reward(self, rewards:List[torch.Tensor], 
                                  discount:float, h:int)->torch.Tensor:
        reward_res = torch.zeros(size=(len(rewards),h))
        for i, r in enumerate(rewards):
            reward = self.get_traj_discnt_reward(
                reward_array=r, discount=discount) 
            reward_res[i,:len(reward)] = reward
        return reward_res

    def get_traj_discnt_reward(self, reward_array:torch.Tensor, 
                               discount:float)->torch.Tensor:
        """ Takes in a tensor of reward values for a trajectory and outputs 
        a tensor of discounted reward values i.e. Tensor([r_{t}*\gamma_{t}])

        Args:
            reward_array (torch.Tensor): Tensor of dimension (traj_length, 1)
            discount (float): One step discount value to apply

        Returns:
            torch.Tensor: Tensor of discounted reward values of dimension 
                (traj_length)
        """
        discnt_tens = torch.Tensor([discount]*len(reward_array))
        discnt_pows = torch.arange(0, len(reward_array))
        discnt_vals = torch.pow(discnt_tens, discnt_pows)
        reward_array = reward_array.squeeze()
        discnt_reward = reward_array*discnt_vals
        return discnt_reward

    def predict_traj_rewards(self, rewards:List[torch.Tensor], 
                             states:List[torch.Tensor], 
                             actions:List[torch.Tensor], weights:torch.Tensor, 
                             discount:float, is_msk:torch.Tensor
                             )->torch.Tensor:
        """_summary_

        Args:
            weights (torch.Tensor): Tensor of IS weights of dimension 
                (# trajectories, max_horizon). Trajectories with length < 
                max_horizon should have zero weight imputed
            discnt_rewards (torch.Tensor): Tensor of discounted rewards  
                of dimension (# trajectories, max_horizon). Trajectories with 
                length < max_horizon should have zero weight imputed
            is_msk (torch.Tensor): Tensor of dimension 
                (# trajectories, max_horizon) defining the lengths of individual 
                trajectories

        Returns:
            torch.Tensor: tensor of size (# trajectories,) defining the 
            individual trajectory rewards
        """
        h:int = weights.shape[1]
        discnt_rewards = self.get_dataset_discnt_reward(
            rewards=rewards, discount=discount, h=h)
        weights = self.process_weights(weights=weights, is_msk=is_msk)
        res:torch.Tensor = torch.mul(discnt_rewards, weights).sum(axis=1)
        return res
    
    