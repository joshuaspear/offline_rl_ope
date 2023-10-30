from abc import ABCMeta, abstractmethod
import numpy as np
import torch
from typing import Callable, List

class Policy(metaclass=ABCMeta):
    
    def __init__(self, policy_class:Callable, collect_res:bool=False, 
                 collect_act:bool=False) -> None:
        self.policy_class = policy_class
        self.policy_predictions = []
        self.policy_actions = []
        if collect_res:
            self.collect_res_fn = self.__cllct_res_true
        else:
            self.collect_res_fn = self.__cllct_false
            
        if collect_act:
            self.collect_act_func =  self.__cllct_act_true
        else:
            self.collect_act_func = self.__cllct_false
        
    def __cllct_false(self, res):
        pass
       
    def __cllct_act_true(self, res):
        self.policy_actions.append(res)
        
    def __cllct_res_true(self, res):
        self.policy_predictions.append(res)

    
    @abstractmethod
    def __call__(self, state:torch.Tensor, action:torch.Tensor)->torch.Tensor:
        """_summary_

        Args:
            state (torch.Tensor): Tensor of dimension (traj_length, state dim)
            action (torch.Tensor): Tensor of dimension 
                (traj_length, number of unique actions). Note, this is likely 
                (traj_length,1) if for example a discrete action space has been 
                flattened from [0,1]^2 to [0,1,2,3]

        Returns:
            torch.Tensor: Tensor of dimension (traj_length, 1), defining the 
            state-action probabilities
        """
        pass

class BehavPolicy(Policy):
    
    def __init__(self, policy_class, collect_res:bool=False, 
                 collect_act:bool=False) -> None:
        super().__init__(policy_class, collect_res=collect_res, 
                         collect_act=collect_act)
        
    def __call__(self, state: torch.Tensor, action: torch.Tensor):
        state = state.detach().numpy()
        action = action.detach().numpy()
        pre_dim = state.shape[0]
        res = self.policy_class.eval_pdf(dep_vals=action, indep_vals=state)
        res = torch.Tensor(res)
        res = res.view(pre_dim, -1)
        self.collect_res_fn(res)
        return res
        

class D3RlPyDeterministic(Policy):
    
    def __init__(
        self, policy_class: Callable, collect_res:bool=False, 
        collect_act:bool=False, gpu:bool=True, eps:float=0
        ) -> None:
        super().__init__(policy_class, collect_res=collect_res, 
                         collect_act=collect_act)
        if gpu:
            self.__preproc_tens = lambda x: x.to("cuda")
            self.__postproc_tens = lambda x: x.to("cpu")
        else:
            self.__preproc_tens = lambda x: x
            self.__postproc_tens = lambda x: x
        self.__eps = eps
        
    
    def __call__(self, state: torch.Tensor, action: torch.Tensor)->torch.Tensor:
        state = self.__preproc_tens(state)
        greedy_action = self.policy_class(x=state).view(-1,1)
        greedy_action = self.__postproc_tens(greedy_action)
        self.collect_act_func(greedy_action)
        res = (greedy_action == action).all(dim=1, keepdim=True).int()
        res_eps_upper = res*(1-self.__eps)
        res_eps_lower = (1-res)*(self.__eps)
        res = res_eps_upper + res_eps_lower
        self.collect_res_fn(res)
        return res
    
class LinearMixedPolicy:
    
    def __init__(self, policy_classes:List[Policy], 
                 mixing_params:torch.Tensor) -> None:
        if sum(mixing_params) != 1:
            raise Exception("Mixing params must equal 1")
        self.__policy_classes = policy_classes
        self.__mixing_params = mixing_params
        self.__policy_predictions = []
    
    @property
    def policy_predictions(self):
        return self.__policy_predictions
        
    def __call__(self, state: torch.Tensor, action: torch.Tensor):
        res = []
        for pol in self.__policy_classes:
            pol_out = pol(state=state, action=action)
            res.append(pol_out)
        res = torch.cat(res, dim=1)
        self.__policy_predictions.append(res)
        return torch.sum(res*self.__mixing_params, dim=1, keepdim=True)