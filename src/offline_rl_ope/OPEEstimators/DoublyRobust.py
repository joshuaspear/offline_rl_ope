import numpy as np
import torch
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
from .IS import ISEstimatorBase
from .DirectMethod import DirectMethodBase



class DREstimator(ISEstimatorBase):
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
        ignore_nan:bool=False, 
        norm_kwargs:Dict[str,Any] = {}
        ) -> None:
        assert isinstance(dm_model,DirectMethodBase)
        assert isinstance(norm_weights,bool)
        assert isinstance(clip,(float,type(None)))
        assert isinstance(cache_traj_rewards,bool)
        assert isinstance(ignore_nan,bool)
        assert isinstance(norm_kwargs,Dict)
        super().__init__(
            norm_weights=norm_weights,
            clip_weights=clip_weights, 
            clip=clip, 
            cache_traj_rewards=cache_traj_rewards, 
            norm_kwargs=norm_kwargs
            )
        self.dm_model = dm_model
        if ignore_nan:
            self.ignore_nan = self.__ignore_nan
        else:
            self.ignore_nan = self.__raise_nan

    @jaxtyped(typechecker=typechecker)
    def __ignore_nan(
        self, 
        p_t:SingleTrajSingleStepTensor
        )->SingleTrajSingleStepTensor:
        # assert isinstance(p_t,torch.Tensor)
        if math.isnan(p_t):
            res = torch.tensor([0.0])
        else:
            res = p_t
        return res

    @jaxtyped(typechecker=typechecker)
    def __raise_nan(
        self, 
        p_t:SingleTrajSingleStepTensor
        )->SingleTrajSingleStepTensor:
        # assert isinstance(p_t,torch.Tensor)
        return p_t

    @jaxtyped(typechecker=typechecker)
    def __update_step(
        self, 
        v_t:SingleTrajSingleStepTensor, 
        p_t:SingleTrajSingleStepTensor,
        r_t:SingleTrajSingleStepTensor, 
        v_dr_t:SingleTrajSingleStepTensor, 
        gamma:SingleTrajSingleStepTensor, 
        q_t:SingleTrajSingleStepTensor
        )->SingleTrajSingleStepTensor:
        """ Predicts the time t+1 doubly robust value prediction based on time
            t values

        Args:
            v_t (torch.Tensor): tensor of size 0, representing the time t state 
                value from a Direct Method
            p_t (torch.Tensor): tensor of size 0, representing the time t 
                importance weight
            r_t (torch.Tensor): tensor of size 0, representing the time t 
                observed reward
            v_dr_t (torch.Tensor): tensor of size 0, representing the time t 
                doubly robust value prediction
            gamma (torch.Tensor): tensor of size 0, representing the time t 
                one step discount factor. Note, this is usually kept constant
            q_t (torch.Tensor): tensor of size 0, representing the time t 
                state-action value from a Direct Method.

        Returns:
            torch.Tensor: tensor of size 0, representing the doubly robust value
                prediction at time t+1
        """
        # check_array_dim(v_t,1)
        # check_array_dim(p_t,1)
        # check_array_dim(r_t,1)
        # check_array_dim(v_dr_t,1)
        # check_array_dim(q_t,1)
        # assert isinstance(v_t,torch.Tensor)
        # assert isinstance(p_t,torch.Tensor)
        # assert isinstance(r_t,torch.Tensor)
        # assert isinstance(v_dr_t,torch.Tensor)
        # assert isinstance(gamma,torch.Tensor)
        # assert isinstance(q_t,torch.Tensor)
        p_t = self.ignore_nan(p_t)
        res = v_t + p_t*(r_t + gamma*v_dr_t - q_t)
        return res
    
    @jaxtyped(typechecker=typechecker)
    def get_traj_discnt_reward(
        self, 
        reward_array:RewardTensor,
        discount:float, 
        state_array:StateTensor, 
        action_array:ActionTensor, 
        weight_array:WeightTensor,
        )->SingleTrajSingleStepTensor:
        """ Takes in a tensor of reward values for a trajectory and outputs 
        a tensor of discounted reward values i.e. Tensor([r_{t}*\gamma_{t}])

        Args:
            reward_array (torch.Tensor): Tensor of dimension (traj_length)
            discount (float): One step discount value to apply
            action_array (torch.Tensor): Tensor of dimension 
            (traj_length, n_actions)
            state_array (torch.Tensor): Tensor of dimension 
            (traj_length, n_states)
            weight_array (torch.Tensor): Tensor of dimension (traj_length, 1)

        Returns:
            torch.Tensor: Tensor of discounted reward values of dimension 
            (traj_length)
        """
        # check_array_dim(reward_array,2)
        # check_array_dim(state_array,2)
        # check_array_dim(action_array,2)
        # check_array_dim(weight_array,2)
        # assert reward_array.shape[1] == 1
        # assert isinstance(reward_array,torch.Tensor)
        # assert isinstance(discount,float)
        # assert isinstance(state_array,torch.Tensor)
        # assert isinstance(action_array,torch.Tensor)
        # assert isinstance(weight_array,torch.Tensor)
        v_dr = torch.tensor([0.0])
        discount = torch.tensor([discount])
        v = self.dm_model.get_v(
            state=state_array
            )
        q = self.dm_model.get_q(
            state=state_array, action=action_array
            )
        reward_array = torch.flip(reward_array, dims=[0])
        v = torch.flip(v, dims=[0])
        q = torch.flip(q, dims=[0])
        weight_array = torch.flip(weight_array, dims=[0])
        for r_t, v_t, q_t, p_t in zip(reward_array,v,q,weight_array):
            v_dr = self.__update_step(
                v_t=v_t, 
                p_t=p_t, 
                r_t=r_t, 
                v_dr_t=v_dr, 
                gamma=discount, 
                q_t=q_t
                )
        return v_dr

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
        l_s = len(states)
        l_r = len(rewards)
        l_a = len(actions)
        l_w = weights.shape[0]
        _msg = f"State({l_s}), rewards({l_r}), actions({l_a}), mask({l_w}) should be equal"
        assert l_s==l_r==l_a==l_w, _msg
        # assert weights.shape == is_msk.shape
        # check_array_dim(weights,2)
        # check_array_dim(is_msk,2)
        weights = self.process_weights(weights=weights, is_msk=is_msk)
        # Tensor looses shape when masked
        weights = torch.masked_select(weights, is_msk > 0)
        weights_lst:List[torch.Tensor] = torch.split(
            weights, is_msk.sum(axis=1).type(torch.int64).tolist())
        reward_res = torch.zeros(size=(len(rewards),))
        for i, (r,s,a,w) in enumerate(
            zip(rewards, states, actions, weights_lst)):
            reward = self.get_traj_discnt_reward(
                reward_array=r, state_array=s, action_array=a, 
                # Reshape weight matrix due to previous mask
                weight_array=w.reshape(-1,1), discount=discount) 
            reward_res[i] = reward
        return reward_res