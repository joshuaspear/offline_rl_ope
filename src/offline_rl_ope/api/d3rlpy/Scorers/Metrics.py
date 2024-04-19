from abc import abstractmethod
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.dataset import ReplayBuffer

from .... import logger
from .base import OPEEstimatorScorerBase
from .IS import ISEstimatorScorer
from ....Metrics import EffectiveSampleSize, MetricBase, ValidWeightsProp

__all__ = [
    "ISMetricScorer", "ValidWeightsPropScorer", "ESSScorer"
    ]

class ISMetricScorer(OPEEstimatorScorerBase):
    
    def __init__(
        self, 
        scorer: ISEstimatorScorer,
        metric: MetricBase
        ) -> None:
        super().__init__(cache=scorer.cache, episodes=None)
        self.scorer = scorer
        self.metric = metric
        
    @abstractmethod
    def __call__(
        self, 
        algo: QLearningAlgoProtocol, 
        dataset: ReplayBuffer
        )->float:
        pass
        
class ValidWeightsPropScorer(ISMetricScorer):
    
    def __init__(
        self, 
        scorer: ISEstimatorScorer, 
        metric: ValidWeightsProp
        ) -> None:
        super().__init__(scorer, metric)
    
    def __call__(
        self, 
        algo: QLearningAlgoProtocol, 
        dataset: ReplayBuffer
        ) -> float:
        weights = self.scorer.process_weights(
                weights=self.cache[self.scorer.is_type].traj_is_weights,
                is_msk=self.cache.weight_msk
            )
        return self.metric(weights, self.cache.weight_msk)

class ESSScorer(ISMetricScorer):
    
    def __init__(
        self, 
        scorer: ISEstimatorScorer, 
        metric: EffectiveSampleSize
        ) -> None:
        super().__init__(scorer, metric)
    
    def __call__(
        self, 
        algo: QLearningAlgoProtocol, 
        dataset: ReplayBuffer
        ) -> float:
        weights = self.scorer.process_weights(
                weights=self.cache[self.scorer.is_type].traj_is_weights,
                is_msk=self.cache.weight_msk
            )
        return self.metric(weights)
        