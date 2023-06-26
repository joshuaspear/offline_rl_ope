from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List
from d3rlpy.metrics.scorer import (AlgoProtocol)
from d3rlpy.dataset import Episode

class OPECallbackBase(metaclass=ABCMeta):
    
    @abstractmethod
    def __call__(self, algo: AlgoProtocol, epoch, total_step):
        pass
    
class QueryCallbackBase(OPECallbackBase):
    
    def __init__(self) -> None:
        self.cache:Dict[str, Any] = {}

    def __getitem__(self, idx:str):
        return self.cache[idx]


class OPEEstimatorScorerBase(metaclass=ABCMeta):
    
    def __init__(self, cache:OPECallbackBase) -> None:
        self.cache = cache
    
    @abstractmethod
    def __call__(self, algo: AlgoProtocol, episodes: List[Episode]):
        pass


    
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
        
    def __call__(self, algo, epoch, total_step) -> Any:
        for i in self.__callbacks:
            i(algo, epoch, total_step)