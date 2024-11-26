from abc import ABCMeta, abstractmethod
import torch
from typing import List
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker

from ..types import WeightTensor
from .EmpiricalMeanDenom import EmpiricalMeanDenomBase

class OPEEstimatorBase(metaclass=ABCMeta):
    
    
    def __init__(
        self, 
        empirical_denom:EmpiricalMeanDenomBase,
        cache_traj_rewards:bool=False
        ) -> None:
        self.traj_rewards_cache:torch.Tensor = torch.Tensor(0)
        if cache_traj_rewards:
            self.__cache_func = self.__cache
        else:
            self.__cache_func = self.__pass_cache
        self.empirical_denom = empirical_denom
    
    def __cache(self, traj_rewards):
        self.traj_rewards_cache = traj_rewards
    
    def __pass_cache(self, traj_rewards):
        pass

    @jaxtyped(typechecker=typechecker)
    def predict(
        self, 
        rewards:List[torch.Tensor], 
        states:List[torch.Tensor], 
        actions:List[torch.Tensor], 
        weights:torch.Tensor, 
        discount:float, 
        is_msk:torch.Tensor
        )->torch.Tensor:
        l_s = len(states)
        l_r = len(rewards)
        l_a = len(actions)
        _msg = f"State({l_s}), rewards({l_r}), actions({l_a}), should be equal"
        assert l_s==l_r==l_a, _msg
        assert isinstance(weights,torch.Tensor)
        assert isinstance(discount,float)
        assert isinstance(is_msk,torch.Tensor)
        traj_rewards = self.predict_traj_rewards(
            rewards=rewards, states=states, actions=actions, weights=weights,
            discount=discount, is_msk=is_msk
            )
        self.__cache_func(traj_rewards)
        denom = self.empirical_denom(
            weights=weights, 
            is_msk=is_msk
        )
        return traj_rewards.sum()/denom
    
    @abstractmethod
    def predict_traj_rewards(
        self, 
        rewards:List[torch.Tensor], 
        states:List[torch.Tensor], 
        actions:List[torch.Tensor], 
        weights:WeightTensor,
        discount:float, 
        is_msk:WeightTensor
        )->Float[torch.Tensor, "n_trajectories"]:
        """Function for subclasses to override defining the trajectory level
        estimates of return

        Args:
            rewards (List[torch.Tensor]): List of Tensors of undiscounted 
                rewards of dimension (max horizon, 1). Trajectories with 
                length < max_horizon should have zero weight imputed
            states (List[torch.Tensor]): List of Tensors of state values. Should 
                be of dimension (traj horizon, state features)
            actions (List[torch.Tensor]): List of Tensors of state values. 
                Should be of dimension (traj horizon, action features)
            weights (torch.Tensor): Tensor of IS weights of dimension 
                (# trajectories, max_horizon). Trajectories with length < 
                max_horizon should have zero weight imputed
            discount (float): One step discount factor
            is_msk (torch.Tensor): Tensor of dimension 
                (# trajectories, max_horizon) defining the lengths of individual 
                trajectories

        Returns:
            torch.Tensor: tensor of size (# trajectories,) defining the 
            individual trajectory rewards
        """
        pass
