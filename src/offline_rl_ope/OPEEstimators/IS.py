from abc import abstractmethod
import torch
from typing import List

from .utils import (wis_norm_weights, norm_weights_pass, clip_weights_pass, 
                    clip_weights)
from .base import OPEEstimatorBase


class ISEstimatorBase(OPEEstimatorBase):
    
    def __init__(self, norm_weights:bool, clip:float=None) -> None:
        super().__init__()
        if norm_weights:
            self.norm_weights = wis_norm_weights
        else:
            self.norm_weights = norm_weights_pass
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

    @abstractmethod
    def predict(self, rewards:List[torch.Tensor], states:List[torch.Tensor], 
                actions:List[torch.Tensor], weights:torch.Tensor, 
                discount:float, is_msk:torch.Tensor
                )->float:
        pass
    

class ISEstimator(ISEstimatorBase):
    
    def __init__(self, norm_weights: bool, clip: float = None) -> None:
        super().__init__(norm_weights, clip)
        
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

    def predict(self, rewards:List[torch.Tensor], states:List[torch.Tensor], 
                actions:List[torch.Tensor], weights:torch.Tensor, 
                discount:float, is_msk:torch.Tensor
                )->float:
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
            float: OPE result
        """
        h:int = weights.shape[1]
        discnt_rewards = self.get_dataset_discnt_reward(
            rewards=rewards, discount=discount, h=h)
        weights = self.process_weights(weights=weights, is_msk=is_msk)
        res:torch.Tensor = torch.mul(discnt_rewards, weights).sum(axis=1)
        res = res.sum()/len(res)
        return res
    
    