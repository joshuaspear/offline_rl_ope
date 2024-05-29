import torch
from typing import Any, Dict, List, Union
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker

from .. import logger
from .utils import (
    WISWeightNorm, VanillaNormWeights, WeightNorm,
    clip_weights_pass as cwp, 
    clip_weights as cw
    )
from .base import OPEEstimatorBase
from ..types import (RewardTensor,WeightTensor)


class ISEstimatorBase(OPEEstimatorBase):
    
    def __init__(
        self, 
        norm_weights:bool, 
        clip_weights:bool=False, 
        cache_traj_rewards:bool=False,
        clip:float=0.0,
        norm_kwargs:Dict[str,Union[str,bool]] = {}
        ) -> None:
        super().__init__(cache_traj_rewards)
        assert isinstance(norm_weights,bool)
        assert isinstance(clip_weights,bool)
        assert isinstance(cache_traj_rewards,bool)
        assert isinstance(clip,float)
        assert isinstance(norm_kwargs,Dict)
        if norm_weights:    
            _norm_weights = WISWeightNorm(**norm_kwargs)
        else:
            _norm_weights = VanillaNormWeights(**norm_kwargs)
        self.norm_weights:WeightNorm = _norm_weights
        self.clip = clip
        if clip_weights:
            self.clip_weights = cw
        else:
            self.clip_weights = cwp
            
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
        # assert isinstance(weights,torch.Tensor)
        # assert isinstance(is_msk,torch.Tensor)
        # assert weights.shape == is_msk.shape
        weights = self.clip_weights(
            traj_is_weights=weights, clip=self.clip)
        weights = self.norm_weights(traj_is_weights=weights, is_msk=is_msk)
        return weights
        

class ISEstimator(ISEstimatorBase):
    
    def __init__(
        self, 
        norm_weights: bool, 
        clip_weights:bool=False, 
        clip: float = 0.0, 
        cache_traj_rewards:bool=False, 
        norm_kwargs:Dict[str,Union[str,bool]] = {}
        ) -> None:
        super().__init__(norm_weights=norm_weights, clip_weights=clip_weights, 
                         clip=clip, cache_traj_rewards=cache_traj_rewards, 
                         norm_kwargs=norm_kwargs)

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
        assert isinstance(discount,float)
        discnt_tens = torch.Tensor([discount]*reward_array.shape[0])
        discnt_pows = torch.arange(0, len(reward_array))
        discnt_vals = torch.pow(discnt_tens, discnt_pows)
        reward_array = reward_array.squeeze()
        discnt_reward = reward_array*discnt_vals
        return discnt_reward
    
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
    
    