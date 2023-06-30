import numpy as np
import torch
from typing import Callable, List, Dict, Tuple, Union, Literal
import math

from .IS import ISEstimatorBase
from .DirectMethod import DirectMethodBase



class DREstimator(ISEstimatorBase):
    """ Doubly robust estimator implemented as per: 
    https://arxiv.org/pdf/1511.03722.pdf
    """
    
    def __init__(self, dm_model:DirectMethodBase, norm_weights: bool, 
                 clip: float = None, ignore_nan:bool=False
                 ) -> None:
        super().__init__(norm_weights=norm_weights, clip=clip)
        self.dm_model = dm_model
        if ignore_nan:
            self.ignore_nan = self.__ignore_nan
        else:
            self.ignore_nan = self.__raise_nan
    
    def __ignore_nan(self, p_t):
        if math.isnan(p_t):
            res = torch.tensor(0)
        else:
            res = p_t
        return res
    
    def __raise_nan(self, p_t):
        return p_t
    
    def __update_step(self, v_t:torch.Tensor, p_t:torch.Tensor, 
                      r_t:torch.Tensor, v_dr_t:torch.Tensor, gamma:torch.Tensor, 
                      q_t:torch.Tensor)->torch.Tensor:
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
        p_t = self.ignore_nan(p_t)
        return v_t + p_t*(r_t + gamma*v_dr_t - q_t)
    
    def get_traj_discnt_reward(self, reward_array:torch.Tensor, 
                               discount:float, state_array:torch.Tensor, 
                               action_array:torch.Tensor, 
                               weight_array:torch.Tensor,
                               )->torch.Tensor:
        """ Takes in a tensor of reward values for a trajectory and outputs 
        a tensor of discounted reward values i.e. Tensor([r_{t}*\gamma_{t}])

        Args:
            reward_array (torch.Tensor): Tensor of dimension (traj_length, 1)
            discount (float): One step discount value to apply

        Returns:
            torch.Tensor: Tensor of discounted reward values of dimension 
            (traj_length)
        """
        v_dr = torch.tensor(0)
        discount = torch.tensor(discount)
        v:torch.Tensor = self.dm_model.get_v(state=state_array)
        q:torch.Tensor = self.dm_model.get_q(
            state=state_array, action=action_array)
        reward_array = torch.flip(reward_array, dims=[0])
        v = torch.flip(v, dims=[0])
        q = torch.flip(q, dims=[0])
        weight_array = torch.flip(weight_array, dims=[0])
        for r_t, v_t, q_t, p_t in zip(reward_array,v,q,weight_array):
            v_dr = self.__update_step(v_t=v_t, p_t=p_t, r_t=r_t, v_dr_t=v_dr, 
                                      gamma=discount, q_t=q_t)
        return v_dr

    def predict(self, rewards:List[torch.Tensor], states:List[torch.Tensor], 
                actions:List[torch.Tensor], weights:torch.Tensor, 
                discount:float, is_msk:torch.Tensor
                )->torch.Tensor:
        """_summary_

        Args:
            rewards (List[torch.Tensor]): _description_
            states (List[torch.Tensor]): _description_
            actions (List[torch.Tensor]): _description_
            weights (torch.Tensor): _description_
            discount (float): _description_
            is_msk (torch.Tensor): _description_

        Returns:
            torch.Tensor: tensor of size 0 defining the mean doubly robust 
            value across the dataset
        """
        test = (
            len(rewards)==len(states)==len(actions)==weights.shape[0]==\
                is_msk.shape[0]
        )
        assert test
        weights = self.process_weights(weights=weights, is_msk=is_msk)
        weights = torch.masked_select(weights, is_msk > 0)
        weights_lst:List[torch.Tensor] = torch.split(
            weights, is_msk.sum(axis=1).type(torch.int64).tolist())
        reward_res = torch.zeros(size=(len(rewards),))
        for i, (r,s,a,w) in enumerate(
            zip(rewards, states, actions, weights_lst)):
            reward = self.get_traj_discnt_reward(
                reward_array=r, state_array=s, action_array=a, 
                weight_array=w, discount=discount) 
            reward_res[i] = reward
        reward_res = reward_res.sum()/len(reward_res)
        return reward_res