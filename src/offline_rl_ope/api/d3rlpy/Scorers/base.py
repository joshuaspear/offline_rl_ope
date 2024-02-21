from typing import Optional, Sequence
from d3rlpy.dataset import EpisodeBase
from d3rlpy.metrics import EvaluatorProtocol

from ..Callbacks.base import OPECallbackBase

class OPEEstimatorScorerBase(EvaluatorProtocol):
    
    def __init__(
        self, 
        cache:OPECallbackBase, 
        episodes: Optional[Sequence[EpisodeBase]] = None
        ) -> None:
        self.cache = cache
        self._episodes = episodes
