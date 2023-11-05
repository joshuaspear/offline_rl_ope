from typing import Sequence, Optional

from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.dataset import EpisodeBase, ReplayBuffer

from .base import OPEEstimatorScorerBase
from ..Callbacks.base import  QueryCallbackBase

__all__ = [
    "QueryScorer"
    ]

class QueryScorer(OPEEstimatorScorerBase):
    
    def __init__(self, cache: QueryCallbackBase, query_key:str, 
                 episodes: Optional[Sequence[EpisodeBase]] = None
                 ) -> None:
        super().__init__(cache=cache, episodes=episodes)
        self.query_key = query_key
        
    def __call__(self, algo: QLearningAlgoProtocol, dataset: ReplayBuffer
                 ) -> float:
        return self.cache[self.query_key]
