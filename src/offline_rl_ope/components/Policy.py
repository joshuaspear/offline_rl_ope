import numpy as np
import torch
from typing import Callable, List

class Policy:
    
    def __init__(self, policy_class:Callable, collect_res:bool=True) -> None:
        self.policy_class = policy_class
        self.policy_predictions = []
        if collect_res:
            self.collect_res_fn = self.__cllct_res_true
        else:
            self.collect_res_fn = self.__cllct_res_false
            
    
    def __cllct_res_true(self, res):
        self.policy_predictions.append(res)
    
    def __cllct_res_false(self, res):
        pass
    
    def __call__(self, state:np.array, action:np.array):
        raise NotImplementedError


class BehavPolicy(Policy):
    
    def __init__(self, policy_class, 
                 collect_res:bool) -> None:
        super().__init__(policy_class, collect_res)
        
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
    
    def __init__(self, policy_class: Callable, collect_res:bool, 
                 collect_act:bool, gpu:bool=True) -> None:
        super().__init__(policy_class, collect_res)
        self.policy_actions = []
        if gpu:
            self.__preproc_tens = lambda x: x.to("cuda")
            self.__postproc_tens = lambda x: x.to("cpu")
        else:
            self.__preproc_tens = lambda x: x
            self.__postproc_tens = lambda x: x
        
        if collect_act:
            self.__collect_act_func =  self.__cllct_act_true
        else:
            self.__collect_act_func = self.__cllct_false
            
    def __cllct_act_true(self, res):
        self.policy_actions.append(res)
        
    def __cllct_false(self, res):
        pass
    
    def __call__(self, state: torch.Tensor, action: torch.Tensor):
        state = self.__preproc_tens(state)
        greedy_action = self.policy_class(x=state).view(-1,1)
        greedy_action = self.__postproc_tens(state)
        self.__collect_act_func(greedy_action)
        res = (greedy_action == action).all(dim=1, keepdim=True).int()
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
        
    
class StableBenchPolicy(Policy):
    
    def __init__(self, policy_class: Callable, collect_res: bool = True, 
                 collect_act_vals: bool = False) -> None:
        super().__init__(policy_class, collect_res)
        self.policy_action_values = []
        if collect_act_vals:
            self.__collect_act_vals_func = self.__cllct_act_vals_true
        else:
            self.__collect_act_vals_func = self.__cllct_false
        
    def __cllct_act_vals_true(self, res):
        self.policy_action_values.append(res)
        
    def __cllct_false(self, res):
        pass
        
    def __call__(self, state: np.array, action: np.array):
        prob = []
        values = []
        for s, a in zip(state, action):
            tmp_res = self.policy_class.evaluate_actions(
                obs=s[None,:], actions=a)
            values.append(tmp_res[0])
            prob.append(torch.exp(tmp_res[1]))
        prob = torch.concat(prob)[:,None]
        values = torch.concat(values)
        self.__collect_act_vals_func(values)
        self.collect_res_fn(prob)
        return prob