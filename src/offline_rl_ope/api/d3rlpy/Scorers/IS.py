import logging
import torch
from typing import Any, Dict, Sequence, Optional
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.dataset import EpisodeBase
from d3rlpy.metrics import EvaluatorProtocol
from d3rlpy.dataset import ReplayBuffer

from ....OPEEstimators import ISEstimator
from .base import OPEEstimatorScorerBase
from ..Callbacks.IS import ISCallback

logger = logging.getLogger("offline_rl_ope")

__all__ = [
    "ISEstimatorScorer", "ISDiscreteActionDistScorer"
    ]


class ISEstimatorScorer(OPEEstimatorScorerBase, ISEstimator):
    
    def __init__(self, discount, cache:ISCallback, is_type:str, 
                 norm_weights: bool, clip_weights:bool=False,
                 clip: float = 0.0, norm_kwargs:Dict[str,Any] = {},
                 episodes:Optional[Sequence[EpisodeBase]] = None
                 ) -> None:
        OPEEstimatorScorerBase.__init__(self, cache=cache, episodes=episodes)
        ISEstimator.__init__(
            self, 
            norm_weights=norm_weights, 
            clip_weights=clip_weights,
            clip=clip, 
            norm_kwargs=norm_kwargs
            )
        self.is_type = is_type
        self.discount = discount
        
    def __call__(
        self, 
        algo: QLearningAlgoProtocol, 
        dataset: ReplayBuffer
        )->float:
        episodes = self._episodes if self._episodes else dataset.episodes
        rewards = [torch.Tensor(ep.rewards) for ep in episodes]
        states = [torch.Tensor(ep.observations) for ep in episodes]
        actions = [torch.Tensor(ep.actions).view(-1,1) for ep in episodes]
        res = self.predict(rewards=rewards, states=states, actions=actions,
                           weights=self.cache[self.is_type].traj_is_weights, 
                           is_msk=self.cache.weight_msk, discount=self.discount
                           )
        return res



class ISDiscreteActionDistScorer(EvaluatorProtocol):
    
    def __init__(self, cache:ISCallback, act:int, 
                 episodes:Optional[Sequence[EpisodeBase]] = None
                 ) -> None:
        self.cache = cache
        self.act = act
        
    def __call__(self, algo: QLearningAlgoProtocol, dataset: ReplayBuffer):
        all_acts = torch.concat(self.cache.policy_actions).squeeze()
        if len(all_acts) == 0:
            logger.warning(
                "Ensure IS Callback object has been set to track policy actions"
                )
        return (all_acts == self.act).sum()/len(all_acts)
        