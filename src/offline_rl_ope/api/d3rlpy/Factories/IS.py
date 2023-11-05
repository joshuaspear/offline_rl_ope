from dataclasses import dataclass
from typing import Any, List, Dict

from d3rlpy.dataset import ReplayBuffer

from ..Callbacks.IS import ISCallback
from ..Scorers.IS import ISEstimatorScorer, ISDiscreteActionDistScorer
from .base import EvaluatorFactoryBase

__all__ = [
    "is_evaluator_factory"
    ]

class IsEvaluatorFactory(EvaluatorFactoryBase):
    
    __est_lkp = {
        "action_dist": ISDiscreteActionDistScorer,
    }
    
    def __call__(
        self, 
        dataset:ReplayBuffer, 
        is_scorers:Dict[str:Dict[str,Any]],
        is_callback:ISCallback,
        unique_action_vals:List[int] = None
        ) -> Dict[str, ISEstimatorScorer]:
        
        scorer_lkp: Dict[str, ISEstimatorScorer] = {}
        
        if "action_dist" in is_scorers:
            assert unique_action_vals is not None
            for act in unique_action_vals:
                scorer_lkp.update({
                    f"action_dist_{act}": self.__est_lkp["action_dist"](
                        cache=is_callback, act=act)
                    })
            del is_scorers["action_dist"]
            
        for scr in is_scorers:
            scorer_lkp.update({
                scr: ISEstimatorScorer(
                    cache=is_callback, episodes=dataset.episodes,
                    **is_scorers[scr]
                    )
                })
        return scorer_lkp
        
is_evaluator_factory = IsEvaluatorFactory()




