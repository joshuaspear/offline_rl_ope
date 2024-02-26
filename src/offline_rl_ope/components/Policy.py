from abc import ABCMeta, abstractmethod
import torch
from typing import Callable, List

def postproc_pass(x:torch.Tensor)->torch.Tensor:
    return x

def postproc_cuda(x:torch.Tensor)->torch.Tensor:
    return x.to("cpu")

def preproc_cuda(x:torch.Tensor)->torch.Tensor:
    return x.to("cuda")


__all__ = [
    "Policy", "GreedyDeterministic", "BehavPolicy"
    ]

class Policy(metaclass=ABCMeta):
    
    def __init__(
        self, 
        policy_func:Callable[..., torch.Tensor], 
        collect_res:bool=False, 
        collect_act:bool=False, 
        gpu:bool=False
        ) -> None:
        """_summary_

        Args:
            policy_func (Callable): Callable that excepts pytorch tensors
            collect_res (bool, optional): _description_. Defaults to False.
            collect_act (bool, optional): _description_. Defaults to False.
        """
        self.policy_func = policy_func
        self.policy_predictions:List[torch.Tensor] = []
        self.policy_actions:List[torch.Tensor] = []
        if collect_res:
            self.collect_res_fn = self.__cllct_res_true
        else:
            self.collect_res_fn = self.__cllct_false
            
        if collect_act:
            self.collect_act_func =  self.__cllct_act_true
        else:
            self.collect_act_func = self.__cllct_false
        if gpu:
            self.preproc_tens = preproc_cuda
            self.postproc_tens = postproc_cuda
        else:
            self.preproc_tens = postproc_pass
            self.postproc_tens = postproc_pass
        
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
    
    def __init__(
        self, 
        policy_func:Callable[[torch.Tensor,torch.Tensor], torch.Tensor], 
        collect_res:bool=False, 
        collect_act:bool=False, 
        gpu:bool=False
        ) -> None:
        super().__init__(policy_func, collect_res=collect_res, 
                         collect_act=collect_act, gpu=gpu)
        
    def __call__(self, state:torch.Tensor, action:torch.Tensor)->torch.Tensor:
        pre_dim = state.shape[0]
        res = self.policy_func(y=action, x=state)
        res = res.view(pre_dim, -1)
        self.collect_res_fn(res)
        return res
        

class GreedyDeterministic(Policy):
    
    def __init__(
        self, 
        policy_func:Callable[[torch.Tensor], torch.Tensor], 
        collect_res:bool=False, 
        collect_act:bool=False, 
        gpu:bool=False, 
        eps:float=0
        ) -> None:
        super().__init__(policy_func, collect_res=collect_res, 
                         collect_act=collect_act, gpu=gpu)
        self.__eps = eps
        
    
    def __call__(self, state: torch.Tensor, action: torch.Tensor)->torch.Tensor:
        state = self.preproc_tens(state)
        greedy_action = self.policy_func(x=state).view(-1,1)
        greedy_action = self.postproc_tens(greedy_action)
        self.collect_act_func(greedy_action)
        res = (greedy_action == action).all(dim=1, keepdim=True).int()
        res_eps_upper = res*(1-self.__eps)
        res_eps_lower = (1-res)*(self.__eps)
        res = res_eps_upper + res_eps_lower
        self.collect_res_fn(res)
        return res
    
