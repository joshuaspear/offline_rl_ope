from typing import List, Dict

from ..Callbacks.Misc import DiscreteValueByActionCallback
from ..Scorers.Misc import QueryScorer
from .base import EvaluatorFactoryBase

__all__ = [
    "dva_evaluator_factory"
    ]

class DvaEvaluatorFactory(EvaluatorFactoryBase):
    
    def __call__(
        self, 
        dva_callback:DiscreteValueByActionCallback,
        unique_action_vals:List[int]
        ) -> Dict[str, QueryScorer]:
        
        scorer_lkp:Dict[str, QueryScorer] = {}
        for act in unique_action_vals:
            scorer_lkp.update({
                    "action_value_{}".format(act): QueryScorer(
                        cache=dva_callback, query_key=act)
                })
        return scorer_lkp
    
dva_evaluator_factory = DvaEvaluatorFactory()