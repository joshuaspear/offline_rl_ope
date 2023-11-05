import logging
import numpy as np
import os
import torch
from typing import Any, Dict, List, Callable
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.dataset import ReplayBuffer

from ....components.Policy import Policy, D3RlPyDeterministic
from ....components.ImportanceSampler import ISWeightOrchestrator
from .base import OPECallbackBase

logger = logging.getLogger("offline_rl_ope")

__all__ = [
    "ISCallback"
    ]


class D3RlPyTorchAlgoPredict:
    
    def __init__(self, predict_func:Callable):
        self.predict_func = predict_func
        
    def __call__(self, x:torch.Tensor):
        pred = self.predict_func(x.cpu().numpy())
        return torch.Tensor(pred)


class ISCallback(ISWeightOrchestrator, OPECallbackBase):
    """Wrapper class for performing importance sampling
    """
    def __init__(
        self, 
        is_types:List[str], 
        behav_policy: Policy, 
        dataset: ReplayBuffer, 
        debug_path:str=None, 
        eval_policy_kwargs=Dict[str,Any] 
        ) -> None:
        OPECallbackBase.__init__(self, debug_path=debug_path)
        ISWeightOrchestrator.__init__(self, *is_types, 
                                      behav_policy=behav_policy)
        self.states:List[torch.Tensor] = []
        self.actions:List[torch.Tensor] = []
        self.rewards:List[torch.Tensor] = []
        for traj in dataset.episodes:
            self.states.append(torch.Tensor(traj.observations))
            self.actions.append(torch.Tensor(traj.actions).view(-1,1))
            self.rewards.append(torch.Tensor(traj.rewards))
        self.eval_policy_kwargs = eval_policy_kwargs

    def debug_true(
        self, 
        algo: QLearningAlgoProtocol, 
        epoch:int, 
        total_step:int
        ):
        for is_type in self.is_samplers.keys():
            np.savetxt(os.path.join(self.debug_path, f"{is_type}_{epoch}.csv"), 
                       self[is_type].traj_is_weights, 
                       delimiter=",")
        
    def run(
        self, 
        algo: QLearningAlgoProtocol, 
        epoch:int, 
        total_step:int
        ) -> None:
        policy_class = D3RlPyTorchAlgoPredict(
            predict_func=algo.predict)
        eval_policy = D3RlPyDeterministic(
            policy_class=policy_class, 
            **self.eval_policy_kwargs
            )
        self.update(states=self.states, actions=self.actions, 
                    eval_policy=eval_policy)