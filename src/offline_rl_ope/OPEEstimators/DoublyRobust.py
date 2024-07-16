import numpy as np
import torch
from torch.nn.functional import pad
from typing import Any, List, Dict
import math
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker

from ..types import (
    WeightTensor, 
    RewardTensor,
    StateTensor,
    ActionTensor,
    SingleTrajSingleStepTensor)
from .IS import ISEstimator
from .DirectMethod import DirectMethodBase
from ..RuntimeChecks import check_array_shape


class DREstimator(ISEstimator):
    """ Doubly robust estimator implemented as per: 
    https://arxiv.org/pdf/1511.03722.pdf
    """
    
    def __init__(
        self, 
        dm_model:DirectMethodBase, 
        norm_weights: bool,
        clip_weights:bool=False,  
        clip:float=0.0, 
        cache_traj_rewards:bool=False, 
        norm_kwargs:Dict[str,Any] = {}
        ) -> None:
        assert isinstance(dm_model,DirectMethodBase)
        assert isinstance(norm_weights,bool)
        assert isinstance(clip,(float,type(None)))
        assert isinstance(cache_traj_rewards,bool)
        assert isinstance(norm_kwargs,Dict)
        super().__init__(
            norm_weights=norm_weights,
            clip_weights=clip_weights, 
            clip=clip, 
            cache_traj_rewards=cache_traj_rewards, 
            norm_kwargs=norm_kwargs
            )
        self.dm_model = dm_model
            
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
        l_s = len(states)
        l_r = len(rewards)
        l_a = len(actions)
        l_w = weights.shape[0]
        _msg = f"State({l_s}), rewards({l_r}), actions({l_a}), mask({l_w}) should be equal"
        assert l_s==l_r==l_a==l_w, _msg
        # discnt_rewards dim is (n_trajectories, max_length)
        h = weights.shape[1]
        n_traj = len(states)
        discnt_rewards = self.get_dataset_discnt_reward(
            rewards=rewards, discount=discount, h=h)
        # weights dim is (n_trajectories, max_length)
        weights = self.process_weights(weights=weights, is_msk=is_msk)
        v:List[Float[torch.Tensor, "max_length 1"]] = []
        q:List[Float[torch.Tensor, "max_length 1"]] = []
        for s,a in zip(states, actions):
            v.append(
                pad(
                    self.dm_model.get_v(state=s), 
                    pad=(0,0,0,h - s.shape[0]), 
                    mode='constant', 
                    value=0
                    )
                )
            q.append(
                pad(
                    self.dm_model.get_q(state=s, action=a), 
                    pad=(0,0,0,h - s.shape[0]), 
                    mode='constant', 
                    value=0
                    )
                )
        v_tens:Float[torch.Tensor, "max_length n_trajectories"] = torch.concat(
            v,dim=1)
        q_tens:Float[torch.Tensor, "max_length n_trajectories"] = torch.concat(
            q,dim=1)
        check_array_shape(v_tens,[h,n_traj])
        check_array_shape(q_tens,[h,n_traj]) 
        v_tens = torch.transpose(v_tens,0,1)
        q_tens = torch.transpose(q_tens,0,1)
        _t1 = torch.mul(discnt_rewards,weights)
        _t2 = torch.mul(q_tens,weights)
        prev_weights = torch.roll(weights,1)
        prev_weights[:,0] = torch.ones(weights.shape[0])
        _t3 = torch.mul(v_tens,prev_weights)
        discnt_vals:Float[
            torch.Tensor, "n_trajectories n_trajectories"
            ] = self.get_discnt_vals(
            discount=discount,
            traj_length=h
            )
        discnt_vals = torch.transpose(
            discnt_vals[:,None].repeat((1,n_traj)),
            0, 1
            )
        _t4 = torch.mul((_t2-_t3),discnt_vals)
        res = (_t1-_t4).sum(dim=1)/n_traj
        return res
        
    
class DR(DREstimator):
    
    def __init__(
        self, 
        dm_model: DirectMethodBase, 
        clip_weights: bool = False, 
        clip: float = 0.0, 
        cache_traj_rewards: bool = False, 
        norm_kwargs: Dict[str, Any] = {}
        ) -> None:
        assert "avg_denom" not in norm_kwargs.keys(), "avg_denom is already set"
        super().__init__(
            dm_model=dm_model, 
            norm_weights=False, 
            clip_weights=clip_weights, 
            clip=clip, 
            cache_traj_rewards=cache_traj_rewards, 
            norm_kwargs={"avg_denom": False, **norm_kwargs}
            )
        
class WDR(DREstimator):
    
    def __init__(
        self, 
        dm_model: DirectMethodBase,  
        clip_weights: bool = False, 
        clip: float = 0.0, 
        cache_traj_rewards: bool = False, 
        norm_kwargs: Dict[str, Any] = {}
        ) -> None:
        assert "avg_denom" not in norm_kwargs.keys(), "avg_denom is already set"
        super().__init__(
            dm_model=dm_model, 
            norm_weights=True, 
            clip_weights=clip_weights, 
            clip=clip, 
            cache_traj_rewards=cache_traj_rewards, 
            norm_kwargs={"avg_denom": False, **norm_kwargs}
            )