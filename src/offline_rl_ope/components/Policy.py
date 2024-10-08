from abc import ABCMeta, abstractmethod
import torch
from typing import Callable, List
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker

from ..RuntimeChecks import check_array_dim
from ..types import (
    StateTensor,
    ActionTensor,
    StateArray,
    ActionArray,
    TorchPolicyReturn,
    NumpyPolicyReturn
    )


def postproc_pass(x:torch.Tensor)->torch.Tensor:
    return x

def postproc_cuda(x:torch.Tensor)->torch.Tensor:
    return x.to("cpu")

def preproc_cuda(x:torch.Tensor)->torch.Tensor:
    return x.to("cuda")


__all__ = [
    "Policy", "GreedyDeterministic", "BasePolicy", "NumpyPolicyFuncWrapper",
    "NumpyGreedyPolicyFuncWrapper"
    ]


class NumpyPolicyFuncWrapper:
    
    def __init__(
        self, 
        policy_func:Callable[[StateArray,ActionArray], NumpyPolicyReturn]
        ) -> None:
        self.policy_func = policy_func
    
    def __call__(
        self,
        state:StateTensor, 
        action:ActionTensor
        )->Float[torch.Tensor, "traj_length 1"]:
        res = self.policy_func(
            state.cpu().detach().numpy(),
            action.cpu().detach().numpy(),
            )
        return res.get_torch_policy_return()
    
class NumpyGreedyPolicyFuncWrapper:
    
    def __init__(
        self, 
        policy_func:Callable[[StateArray], NumpyPolicyReturn]
        ) -> None:
        self.policy_func = policy_func
    
    def __call__(
        self,
        state:StateTensor 
        )->ActionTensor:
        res = self.policy_func(
            state.cpu().detach().numpy()
            )
        return res.get_torch_policy_return()


class BasePolicy(metaclass=ABCMeta):
    
    def __init__(
        self, 
        policy_func:Callable[...,TorchPolicyReturn], 
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
    def __call__(
        self, 
        state:StateTensor, 
        action:ActionTensor
        )->Float[torch.Tensor, "traj_length 1"]:
        """Defines the probability of the given actions under the given states
        according to the policy defined by policy_func

        Args:
            state (torch.Tensor): Tensor of dimension (traj_length, state dim)
            action (torch.Tensor): Tensor of dimension 
                (traj_length, number of unique actions). Note, this is likely 
                (traj_length,1) if for example a discrete action space has been 
                flattened from [0,1]^2 to [0,1,2,3]

        Returns:
            torch.Tensor: Tensor of dimension (traj_length, 1), defining the 
            state-action probabilities i.e., if the action space is 
            n-dimensional, the output probability is the joint over the actions
        """
        pass

class Policy(BasePolicy):
    
    def __init__(
        self, 
        policy_func:Callable[
            [StateTensor,ActionTensor], 
            TorchPolicyReturn
            ], 
        collect_res:bool=False, 
        collect_act:bool=False, 
        gpu:bool=False
        ) -> None:
        super().__init__(policy_func, collect_res=collect_res, 
                         collect_act=collect_act, gpu=gpu)
        
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        state:StateTensor, 
        action:ActionTensor
        )->Float[torch.Tensor, "traj_length 1"]:
        # assert isinstance(state,torch.Tensor)
        # assert isinstance(action,torch.Tensor)
        # check_array_dim(action,2)
        state = self.preproc_tens(state)
        action = self.preproc_tens(action)
        p_return = self.policy_func(state, action)
        actions = self.postproc_tens(p_return.actions)
        action_prs = self.postproc_tens(p_return.action_prs)
        self.collect_res_fn(action_prs)
        self.collect_act_func(actions)
        return action_prs

class GreedyDeterministic(BasePolicy):
    
    def __init__(
        self, 
        policy_func:Callable[[StateTensor], TorchPolicyReturn], 
        collect_res:bool=False, 
        collect_act:bool=False, 
        gpu:bool=False, 
        eps:float=0
        ) -> None:
        super().__init__(policy_func, collect_res=collect_res, 
                         collect_act=collect_act, gpu=gpu)
        self.__eps = eps
        
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        state:StateTensor, 
        action:ActionTensor
        )->Float[torch.Tensor, "traj_length 1"]:
        # assert isinstance(state,torch.Tensor)
        # assert isinstance(action,torch.Tensor)
        # check_array_dim(action,2)
        state = self.preproc_tens(state)
        p_return = self.policy_func(state)
        greedy_action = p_return.actions
        check_array_dim(greedy_action,2)
        assert action.shape == greedy_action.shape
        greedy_action = self.postproc_tens(greedy_action)
        self.collect_act_func(greedy_action)
        res = (greedy_action == action).all(dim=1, keepdim=True).float()
        res_eps_upper = res*(1-self.__eps)
        res_eps_lower = (1-res)*(self.__eps)
        res = res_eps_upper + res_eps_lower
        self.collect_res_fn(res)
        return res
