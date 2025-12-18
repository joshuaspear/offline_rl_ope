import torch
from typing import Any, Dict, List, Union
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker
from abc import ABCMeta, abstractmethod

from .utils import (
    clip_weights_pass as cwp, 
    clip_weights as cw
    )
from .EmpiricalMeanDenom import EmpiricalMeanDenomBase
from .WeightDenom import WeightDenomBase
from ..types import (RewardTensor,WeightTensor)


class ISEstimatorBase(metaclass=ABCMeta):
    
    def __init__(
        self, 
        empirical_denom:EmpiricalMeanDenomBase,
        weight_denom:WeightDenomBase, 
        clip_weights:bool=False, 
        cache_traj_rewards:bool=False,
        clip:float=0.0,
        ) -> None:
        assert isinstance(weight_denom,WeightDenomBase)
        assert isinstance(clip_weights,bool)
        assert isinstance(cache_traj_rewards,bool)
        assert isinstance(clip,float)
        self.traj_rewards_cache:torch.Tensor = torch.Tensor(0)
        if cache_traj_rewards:
            self.__cache_func = self.__cache
        else:
            self.__cache_func = self.__pass_cache
        self.empirical_denom = empirical_denom
        self.clip = clip
        if clip_weights:
            self.clip_weights = cw
        else:
            self.clip_weights = cwp
        self.weight_denom = weight_denom
        
    def __cache(self, traj_rewards):
        self.traj_rewards_cache = traj_rewards
    
    def __pass_cache(self, traj_rewards):
        pass
            
    @jaxtyped(typechecker=typechecker)
    def process_weights(
        self, 
        weights:WeightTensor, 
        is_msk:WeightTensor
        )->WeightTensor:
        """Processes one step weights (i.e., clipping and weighting)

        Args:
            weights (WeightTensor): Tensor of one step, unprocessed weights of 
            dimension (n_trajectories, max_length)
            is_msk (WeightTensor): Mask tensor for weights of dimension 
            (n_trajectories, max_length)

        Returns:
            WeightTensor: Tensor of processed weight, of dimension 
            (n_trajectories, max_length)
        """
        weights = self.clip_weights(weights=weights, clip=self.clip)
        weights = self.weight_denom(weights=weights, is_msk=is_msk)
        return weights
    
    def get_dataset_discnt_reward(
        self, 
        rewards:List[torch.Tensor],
        discount:float, 
        h:int
        )->Float[torch.Tensor, "n_trajectories max_length"]:
        """Loops over a dataset of trajectories of undiscounted rewards and 
        calculates the discounted rewards

        Args:
            rewards (List[torch.Tensor]): List of undiscounted trajectories of 
            rewards, each with dimension (traj_length,1)
            discount (float): One step discount value to apply
            h (int): The maximum trajectory length (max_length)
        
        Returns:
            torch.Tensor: Returns tensor of one step discounted reward values
        """
        assert isinstance(discount,float)
        assert isinstance(h,int)
        reward_res = torch.zeros(size=(len(rewards),h))
        for i, r in enumerate(rewards):
            # Output dim is (traj_length,)
            reward = self.get_traj_discnt_reward(
                reward_array=r, discount=discount) 
            reward_res[i,:len(reward)] = reward
        return reward_res

    @jaxtyped(typechecker=typechecker)
    def get_discnt_vals(
        self, 
        discount:float,
        traj_length:int
        ):
        discnt_tens = torch.Tensor([discount]*traj_length)
        discnt_pows = torch.arange(0, traj_length)
        discnt_vals = torch.pow(discnt_tens, discnt_pows)
        return discnt_vals
    
    @jaxtyped(typechecker=typechecker)
    def get_traj_discnt_reward(
        self, 
        reward_array:RewardTensor,
        discount:float
        )->Float[torch.Tensor, "traj_length"]:
        """ Takes in a tensor of reward values for a trajectory and outputs 
        a tensor of discounted reward values i.e. Tensor([r_{t}*\gamma_{t}])

        Args:
            reward_array (torch.Tensor): Tensor of dimension (traj_length, 1)
            discount (float): One step discount value to apply

        Returns:
            torch.Tensor: Tensor of discounted reward values of dimension 
                (traj_length)
        """
        #assert reward_array.shape[1] == 1
        #assert isinstance(reward_array,torch.Tensor)
        discnt_vals = self.get_discnt_vals(
            discount=discount, traj_length=reward_array.shape[0]
        )
        reward_array = reward_array.squeeze()
        discnt_reward = reward_array*discnt_vals
        return discnt_reward

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
        weights = self.clip_weights(weights=weights, clip=self.clip)
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


class ISEstimator(ISEstimatorBase):
    
    def __init__(
        self, 
        empirical_denom:EmpiricalMeanDenomBase,
        weight_denom: WeightDenomBase, 
        clip_weights:bool=False, 
        clip: float = 0.0, 
        cache_traj_rewards:bool=False
        ) -> None:
        super().__init__(
            empirical_denom=empirical_denom,
            weight_denom=weight_denom,
            clip_weights=clip_weights,
            cache_traj_rewards=cache_traj_rewards,
            clip=clip
            )
    
    @jaxtyped(typechecker=typechecker)
    def predict_traj_rewards(
        self, 
        rewards:List[torch.Tensor], 
        states:List[torch.Tensor], 
        actions:List[torch.Tensor], 
        weights:WeightTensor,
        discount:float, 
        is_msk:WeightTensor
        )->Float[torch.Tensor, "n_trajectories"]:
        """Main calculation function:
            1. Loops over a dataset of trajectories of undiscounted rewards;
            2. Processes one step weights (i.e., clipping and weighting)
            3. Calculates the individual trajectory reward

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
        l_r = len(rewards)
        l_w = weights.shape[0]
        _msg = f"rewards({l_r}), mask({l_w}) should be equal"
        assert l_r==l_w, _msg
        h = weights.shape[1]
        # discnt_rewards dim is (n_trajectories, max_length)
        discnt_rewards = self.get_dataset_discnt_reward(
            rewards=rewards, discount=discount, h=h)
        # weights dim is (n_trajectories, max_length)
        weights = self.process_weights(weights=weights, is_msk=is_msk)
        # (n_trajectories,max_length) ELEMENT WISE * (n_trajectories,max_length)
        res = torch.mul(discnt_rewards,weights).sum(dim=1)
        return res