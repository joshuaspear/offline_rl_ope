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
from ..Misc import D3RlPyDeterministicWrapper
from ..Policy import PolicyFactory

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
        policy_factory:PolicyFactory,
        debug:bool=False,
        debug_path:str="", 
        ) -> None:
        OPECallbackBase.__init__(self, debug=debug, debug_path=debug_path)
        ISWeightOrchestrator.__init__(self, *is_types, 
                                      behav_policy=behav_policy)
        self.states:List[torch.Tensor] = []
        self.actions:List[torch.Tensor] = []
        self.rewards:List[torch.Tensor] = []
        self.policy_factory = policy_factory
        for traj in dataset.episodes:
            self.states.append(torch.Tensor(traj.observations))
            self.actions.append(torch.Tensor(traj.actions))
            self.rewards.append(torch.Tensor(traj.rewards))
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
        eval_policy = self.policy_factory.create(algo=algo)
        self.update(states=self.states, actions=self.actions, 
                    eval_policy=eval_policy)