from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Optional, Sequence
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.dataset import EpisodeBase
from d3rlpy.metrics import EvaluatorProtocol

class OPECallbackBase(metaclass=ABCMeta):
    
    @abstractmethod
    def __call__(self, algo: QLearningAlgoProtocol,  epoch:int, total_step:int):
        pass
    
class QueryCallbackBase(OPECallbackBase):
    
    def __init__(self) -> None:
        self.cache:Dict[str, Any] = {}

    def __getitem__(self, idx:str):
        return self.cache[idx]


class OPEEstimatorScorerBase(EvaluatorProtocol):
    
    def __init__(self, cache:OPECallbackBase, 
                 episodes: Optional[Sequence[EpisodeBase]] = None
                 ) -> None:
        self.cache = cache
        self._episodes = episodes

    
# class WrapperAccessor:
#     """ Helper class for accessing results stored in the cache of a 'Wrapper'
#     object
#     """
#     def __init__(self, item_key, wrapper:Wrapper) -> None:
#         self.__wrapper = wrapper
#         self.__item_key = item_key
        
#     def __call__(self, *args, **kwargs):
#         return self.__wrapper[self.__item_key]
    
class EpochCallbackHandler:
    """Helper class for executing multiple wrapper objectes with a single calls
    """
    def __init__(self, callbacks:List[OPECallbackBase]) -> None:
        self.__callbacks = callbacks
        
    def __call__(self, algo:QLearningAlgoProtocol,  epoch:int, total_step:int
                 ) -> Any:
        for i in self.__callbacks:
            i(algo, epoch, total_step)