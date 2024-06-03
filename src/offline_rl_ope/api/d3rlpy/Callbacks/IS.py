import logging
import numpy as np
import os
import torch
from typing import Any, Dict, List
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.dataset import ReplayBuffer

from ....components.Policy import BasePolicy, GreedyDeterministic
from ....components.ImportanceSampler import ISWeightOrchestrator
from .base import OPECallbackBase
from ..Misc import D3RlPyTorchAlgoPredict

logger = logging.getLogger("offline_rl_ope")

__all__ = [
    "ISCallback"
    ]

class ISCallback(ISWeightOrchestrator, OPECallbackBase):
    """Wrapper class for performing importance sampling
    """
    def __init__(
        self, 
        is_types:List[str], 
        behav_policy: BasePolicy, 
        dataset: ReplayBuffer,
        action_dim:int, 
        eval_policy_kwargs:Dict[str,Any] = {},
        debug:bool=False,
        debug_path:str="", 
        ) -> None:
        OPECallbackBase.__init__(self, debug=debug, debug_path=debug_path)
        ISWeightOrchestrator.__init__(self, *is_types, 
                                      behav_policy=behav_policy)
        self.states:List[torch.Tensor] = []
        self.actions:List[torch.Tensor] = []
        self.rewards:List[torch.Tensor] = []
        self.action_dim = action_dim
        for traj in dataset.episodes:
            self.states.append(torch.Tensor(traj.observations))
            self.actions.append(torch.Tensor(traj.actions))
            self.rewards.append(torch.Tensor(traj.rewards))
        self.eval_policy_kwargs = eval_policy_kwargs
        assert len(self.states[0].shape) == 2
        assert len(self.actions[0].shape) == 2
        assert self.rewards[0].shape[1] == 1

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
        policy_func = D3RlPyTorchAlgoPredict(
            predict_func=algo.predict,
            action_dim=self.action_dim
            )
        eval_policy = GreedyDeterministic(
            policy_func=policy_func, 
            **self.eval_policy_kwargs
            )
        self.update(states=self.states, actions=self.actions, 
                    eval_policy=eval_policy)