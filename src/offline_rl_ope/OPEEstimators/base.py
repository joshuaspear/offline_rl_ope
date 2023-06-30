from abc import ABCMeta, abstractmethod
import torch
from typing import List


class OPEEstimatorBase(metaclass=ABCMeta):
    
    
    def __init__(self, cache_traj_rewards:bool=False) -> None:
        self.traj_rewards_cache:torch.Tensor = torch.Tensor(0)
        if cache_traj_rewards:
            self.__cache_func = self.__cache
        else:
            self.__cache_func = self.__pass_cache
    
    def __cache(self, traj_rewards):
        self.traj_rewards_cache = traj_rewards
    
    def __pass_cache(self, traj_rewards):
        pass
    
    def predict(self, rewards:List[torch.Tensor], states:List[torch.Tensor], 
                actions:List[torch.Tensor], weights:torch.Tensor, 
                discount:float, is_msk:torch.Tensor
                )->torch.Tensor:
        traj_rewards = self.predict_traj_rewards(
            rewards=rewards, states=states, actions=actions, weights=weights,
            discount=discount, is_msk=is_msk
            )
        self.__cache_func(traj_rewards)
        return traj_rewards.mean()
    
    @abstractmethod
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
        pass
