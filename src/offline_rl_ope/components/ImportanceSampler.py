from abc import ABCMeta, abstractmethod
import logging
from typing import Dict, List, Tuple
import torch
from torch.nn.functional import pad

from .Policy import Policy


logger = logging.getLogger("offline_rl_ope")


class ISWeightCalculator:
    def __init__(self, behav_policy:Policy) -> None:
        self.__behav_policy = behav_policy
        self.is_weights = torch.empty(0)
        self.weight_msk = torch.empty(0)
        self.policy_actions = torch.empty(0)
        
    def get_traj_w(self, states:torch.Tensor, actions:torch.Tensor, 
                   eval_policy:Policy)->torch.Tensor:
        """Function to calculate the timestep IS weights over a trajectory i.e., 
        for each timestep (t) Tensor(\pi_{e}(a_{t}|s_{t})/\pi_{b}(a_{t}|s_{t}))
        Args:
            states (torch.Tensor): Tensor of dimension (traj_length, state size)
            actions (torch.Tensor): Tensor of dimension 
                (traj_length, action size)

        Returns:
            torch.Tensor: Tensor of dimension (traj_length) defining the 
            propensity weights for the input trajectory
        """
        if (len(states.shape) != 2) | (len(actions.shape) != 2):
            logger.debug("states.shape: {}".format(states.shape))
            logger.debug("actions.shape: {}".format(actions.shape))
            raise Exception("State and actions should have 2 dimensions")
        with torch.no_grad():
            behav_probs = self.__behav_policy(action=actions, 
                                            state=states)
            #logger.debug("behav_probs: {}".format(behav_probs))
            eval_probs = eval_policy(action=actions, state=states)
        #logger.debug("eval_probs: {}".format(eval_probs))
        weight_array = eval_probs/behav_probs
        weight_array = weight_array.view(-1)
        #logger.debug("weight_array: {}".format(weight_array))
        return weight_array
    
    def get_dataset_w(self, states:List[torch.Tensor], 
                      actions:List[torch.Tensor], 
                      eval_policy:Policy
                      )->Tuple[torch.Tensor, torch.Tensor]:
        """_summary_

        Args:
            states (List[torch.Tensor]): List of input Tensors of dimensions
            (traj_length, number of states)
            actions (List[torch.Tensor]): List of input Tensors of dimensions
                (traj_length, number of actions). Note, this is likely 
                (traj_length,1) if for example a discrete action space has been 
                flattened from [0,1]^2 to [0,1,2,3]
            eval_policy (Policy): Policy class defining the target policy to be 
                evaluated

        Returns:
            Tuple[torch.Tensor]: Tuple of tensors where:
                weight_res is a (# trajectories, max(traj_length)) Tensor. 
                weight_res[i,j] defines the jth timestep propensity weight for 
                the ith trajectory
                weight_msk is a (# trajectories, max(traj_length)) binary 
                Tensor. weight_msk[i,j] defines whether the jth timestep of the
                ith trajectory was observed
        """
        assert len(states) == len(actions)
        # weight_res = torch.zeros(size=(len(states),h))
        # weight_msk = torch.zeros(size=(len(states),h))
        weight_res_lst:List[torch.Tensor] = []
        weight_msk_lst:List[torch.Tensor] = []
        h = 0
        for i, (s,a) in enumerate(zip(states, actions)):
            weight = self.get_traj_w(states=s, actions=a, 
                                     eval_policy=eval_policy) 
            #weight_res_lst[i,:len(weight)] = weight
            weight_res_lst.append(weight)
            __h = len(weight)
            #weight_msk_lst[i,:len(weight)] = 
            weight_msk_lst.append(torch.ones(size=[__h]))
            h = max(h,__h)
        # pad all tensors to have same length
        weight_res_lst = [
            pad(x, pad=(0, h - x.numel()), mode='constant', value=0) 
            for x in weight_res_lst
            ]
        weight_res = torch.stack(weight_res_lst)
        # stack them
        weight_msk_lst = [
            pad(x, pad=(0, h - x.numel()), mode='constant', value=0) 
            for x in weight_msk_lst
            ]
        weight_msk = torch.stack(weight_msk_lst)
        return weight_res, weight_msk
    
    def update(self, states:List[torch.Tensor], actions:List[torch.Tensor], 
               eval_policy:Policy):
        self.is_weights, self.weight_msk = self.get_dataset_w(
            states=states, actions=actions, eval_policy=eval_policy)
        self.policy_actions = eval_policy.policy_actions

    def flush(self):
        self.is_weights = torch.empty(0)
        

class ImportanceSampler(metaclass=ABCMeta):
    
    def __init__(self, is_weight_calc:ISWeightCalculator) -> None:
        self.is_weight_calc = is_weight_calc
        self.traj_is_weights = torch.empty(0)
        
    
    def update(self):
        self.traj_is_weights = self.get_traj_weight_array(
            is_weights=self.is_weight_calc.is_weights,
            weight_msk=self.is_weight_calc.weight_msk
        )
    
    def flush(self):
        self.traj_is_weights = torch.empty(0)
    
    @abstractmethod
    def get_traj_weight_array(
        self, is_weights:torch.Tensor, weight_msk:torch.Tensor
        )->torch.Tensor:
        """Performs additional calculations on the weights i.e. for per 
        decision. Weights are products up to the current timestep

        Args:
            is_weights (torch.Tensor): Tensor of dimension 
                (# trajectories, max(traj_length))
            weight_msk (torch.Tensor): Tensor of dimension 
                (# trajectories, max(traj_length))
            
        Returns: 
            torch.Tensor: Transformed weight array of dimension 
                (# trajectories, max(traj_length))
        """
        pass
        
    
class VanillaIS(ImportanceSampler):
        
    def get_traj_weight_array(
        self, 
        is_weights:torch.Tensor, 
        weight_msk:torch.Tensor
        )->torch.Tensor:
        __orig_dim = is_weights.shape
        # Convert missing timesteps in trajectories from 0 to 1 otherwise prod
        # will be 0 for trjectories without the max number of timesteps
        is_weights[weight_msk==0] = 1
        is_weights = torch.prod(is_weights, dim=1, keepdim=True)
        is_weights = is_weights.expand(__orig_dim)
        # Convert missing timesteps in trajectories back to 0
        is_weights = is_weights*weight_msk
        return is_weights

class PerDecisionIS(ImportanceSampler):
    
    def get_traj_weight_array(
        self, 
        is_weights:torch.Tensor,
        weight_msk:torch.Tensor
        ):
        is_weights = torch.cumprod(is_weights, dim=1)
        is_weights = is_weights*weight_msk
        return is_weights


class ISWeightOrchestrator(ISWeightCalculator):
    
    __is_fact_lkp = {
        "vanilla": VanillaIS,
        "per_decision": PerDecisionIS
    }
    
    def __init__(self, *args, behav_policy:Policy) -> None:
        super().__init__(behav_policy=behav_policy)
        self.is_samplers:Dict[str,ImportanceSampler] = {}
        for arg in args:
            try:
                self.is_samplers[arg] = self.__is_fact_lkp[arg](
                    is_weight_calc=self)
            except KeyError as e:
                logger.error("Accepted samplers are: {}".format(
                    list(self.__is_fact_lkp.keys())))
                
    def __len__(self):
            return len(self.is_samplers)

    def __getitem__(self, idx):
        return self.is_samplers[idx]
    
    def update(self, states:List[torch.Tensor], actions:List[torch.Tensor], 
               eval_policy:Policy):
        super().update(states=states, actions=actions, eval_policy=eval_policy)
        for sampler in self.is_samplers.keys():
            self.is_samplers[sampler].update()