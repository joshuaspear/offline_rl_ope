from typing import Dict

from ..Scorers.Misc import QueryScorer
from .base import EvaluatorFactoryBase

__all__ = [
    "fqe_evaluator_factory"
]

class FqeEvaluatorFactory(EvaluatorFactoryBase):
    
    def __call__(
        self,
        fqe_scorers,
        fqe_callback
        ) -> Dict[str, QueryScorer]:
        
        scorer_lkp: Dict[str, QueryScorer] = {}
        
        for scr in fqe_scorers:
            scorer_lkp.update(
                {scr: QueryScorer(cache=fqe_callback, query_key=scr)}
                )
        return scorer_lkp


fqe_evaluator_factory = FqeEvaluatorFactory()
