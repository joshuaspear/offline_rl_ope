import logging
from typing import Dict, List, Tuple
import numpy as np
import torch

from .Policy import Policy

logger = logging.getLogger("offline_rl_ope")

class ImportanceSampling:
    
    def __init__(self, behav_policy:Policy, eval_policy:Policy, 
                 discount:float, norm_weights:bool=False) -> None:
        self.__behav_policy = behav_policy
        self.__eval_policy = eval_policy
        self.__discount = discount
        self.is_indiv_weights = []
        self.is_weights = []
        self.norm_weights = norm_weights
    
    def __eval_array_weight(self, state_array:torch.Tensor, 
                            action_array:torch.Tensor)->torch.Tensor:
        """Function to calculate the timestep IS weights over a trajectory i.e., 
        for each timestep (t) Tensor(\pi_{e}(a_{t}|s_{t})/\pi_{b}(a_{t}|s_{t}))
        Args:
            state_array (torch.Tensor): Tensor of dimension (traj_length, state size)
            action_array (torch.Tensor): Tensor of dimension (traj_length, action size)

        Returns:
            torch.Tensor: Tensor defining the timestep IS weights of dimension (traj_length)
        """
        behav_probs = self.__behav_policy(action=action_array, 
                                          state=state_array)
        logger.debug("behav_probs: {}".format(behav_probs))
        eval_probs = self.__eval_policy(action=action_array, state=state_array)
        logger.debug("eval_probs: {}".format(eval_probs))
        weight_array = eval_probs/behav_probs
        weight_array = weight_array.squeeze()
        logger.debug("weight_array: {}".format(weight_array))
        norm_cont = torch.prod(weight_array)
        return weight_array, norm_cont
    
    def __eval_traj_reward(self, reward_array:torch.Tensor)->torch.Tensor:
        """ Takes in a tensor of reward values for a trajectory and outputs 
        a tensor of discounted reward values i.e. Tensor([r_{t}, \gamma_{t}])

        Args:
            reward_array (torch.Tensor): Tensor of dimension (traj_length, 1)

        Returns:
            torch.Tensor: Tensor of discounted reward values of dimension (traj_length)
        """
        discnt_tens = torch.Tensor([self.__discount]*len(reward_array))
        discnt_pows = torch.arange(0, len(reward_array))
        discnt_vals = torch.pow(discnt_tens, discnt_pows)
        reward_array = reward_array.squeeze()
        discnt_reward = reward_array*discnt_vals
        return discnt_reward    
        
    def get_traj_weight_array(self, weight_array:torch.Tensor)->torch.Tensor:
        """Performs additional calculations on the weights i.e. for per 
        decision. Weights are products up to the current timestep

        Args:
            weight_array (torch.Tensor): Tensor of dimension (traj_length)

        Raises:
            NotImplementedError: Expected implementation in child class
            
        Returns: 
            torch.Tensor: Transformed weight array
        """
        raise NotImplementedError        
    
    def get_traj_w_r(self, state:torch.Tensor, action:torch.Tensor, 
                      reward:torch.Tensor)->Tuple[torch.Tensor]:
        """ Calculates the trejectory level weights and discounted rewards

        Args:
            state (torch.Tensor): Tensor of dimension (traj_length, state size)
            action (torch.Tensor): Tensor of dimension (traj_length, action size)
            reward (torch.Tensor): Tensor of dimension (traj_length, 1)

        Returns:
            Tuple[torch.Tensor]: Tuples of torch tensors with dimension ((1),(1))
        """
        weight_array, norm_cont = self.__eval_array_weight(state_array=state, 
                                                           action_array=action)
        discnt_reward = self.__eval_traj_reward(reward_array=reward)
        weight_array = self.get_traj_weight_array(weight_array=weight_array)
        return weight_array ,discnt_reward, norm_cont
        
    def get_dataset_w_r(self, dataset:List[Dict[str,np.array]]
                        )->List[torch.Tensor]:
        weight_res = []
        discnt_reward_res = []
        norm_conts = []
        # For each trajectory in an input dataset, get a list of the weights
        # and rewards for each trajectory
        for i, vals in enumerate(dataset):
            weight, discnt_reward, norm_cont = self.get_traj_w_r(
                state=torch.Tensor(vals["state"]), 
                action=torch.Tensor(vals["act"]), 
                reward=torch.Tensor(vals["reward"])
                )
            weight_res.append(weight)
            discnt_reward_res.append(discnt_reward)
            norm_conts.append(norm_cont)
        return weight_res, discnt_reward_res, norm_conts

    
    
class VanillaIS(ImportanceSampling):
    
    def __init__(self, behav_policy:Policy, eval_policy:Policy, 
                 discount:float) -> None:
        super().__init__(behav_policy=behav_policy, eval_policy=eval_policy, 
                         discount=discount)
        
    def get_traj_weight_array(self, weight_array):
        self.is_indiv_weights.append(weight_array)
        __orig_len = len(weight_array)
        weight_array = torch.prod(weight_array, dim=0, keepdim=True)
        weight_array = weight_array.expand([__orig_len])
        self.is_weights.append(weight_array)
        return weight_array

class PerDecisionIS(ImportanceSampling):
    
    def __init__(self, behav_policy: Policy, eval_policy: Policy, 
                 discount: float) -> None:
        super().__init__(behav_policy, eval_policy, discount)
    
    def get_traj_weight_array(self, weight_array):
        weight_array = torch.cumprod(weight_array, dim=0)
        self.is_indiv_weights.append(weight_array)
        return weight_array        

# Implemented as per https://arxiv.org/pdf/2105.02580.pdf
# class ISTimeDep:

#     def __init__(self, behav_policy:Policy, eval_policy:Policy, 
#                  discount_b:float, discount_tau:float) -> None:
#         self.__behav_policy = behav_policy
#         self.__eval_policy = eval_policy
#         self.discount_b = discount_b
#         self.discount_tau = discount_tau
#         self.is_indiv_weights = []
#         self.is_weights = []
#         self.discount_values = []
        
    
#     def __eval_array_weight(self, state_array:np.array, 
#                             action_array:np.array)->np.array:
#         """_summary_

#         Args:
#             state (np.array): Array[batch_no, num_state_features]
#             action (np.array): Array[batch_no, num_action_features]

#         Returns:
#             _type_: _description_
#         """
#         behav_probs = self.__behav_policy(action=action_array, 
#                                           state=state_array)
#         eval_probs = self.__eval_policy(action=action_array, state=state_array)
#         weight_array = eval_probs/behav_probs
#         return weight_array
    
#     def __eval_traj_reward(self, reward_array:torch.Tensor, 
#                            time_diffs:torch.Tensor)->torch.Tensor:
#         discnt_vals = self.discount_b^(time_diffs/self.discount_tau)
#         self.discount_values.append(discnt_vals)
#         reward_array = reward_array.squeeze()
#         discnt_reward = reward_array*discnt_vals
#         discnt_reward = sum(discnt_reward)
#         return discnt_reward
        
    
#     def get_traj_loss(self, state:torch.Tensor, action:torch.Tensor, 
#                       reward:torch.Tensor, time_diffs:torch.Tensor
#                       )-> torch.Tensor:
#         weight_array = self.__eval_array_weight(state_array=state, 
#                                                 action_array=action)
#         self.is_indiv_weights.append(weight_array)
#         weight = torch.prod(weight_array)
#         self.is_weights.append(weight)
#         discnt_reward = self.__eval_traj_reward(reward_array=reward, 
#                                                 time_diffs=time_diffs)
#         return weight*discnt_reward
    
#     def get_batch_traj_loss(self, state:List[torch.Tensor], 
#                             action:List[torch.Tensor], 
#                             reward:List[torch.Tensor], 
#                             time_diffs:List[torch.Tensor]):
#         res = []
#         for s,a,r,t in zip(state, action, reward, time_diffs):
#             res.append(self.get_traj_loss(state=s, action=a, reward=r, 
#                                           time_diffs=t))
#         return res