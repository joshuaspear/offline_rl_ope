from typing import Any, Dict, List
from d3rlpy.metrics.scorer import (AlgoProtocol)

class Wrapper:
    """Base class for building OPE methods with cached results. After each call
    to the object, the cache is refreshed and made available for querying
    """
    def __init__(self, scorers_nms:List[str]) -> None:
        """Object init

        Args:
            scorers_nms (List[str]): List of unique results produced by 'eval' 
            and stord in cache
        """
        self.scorers_nms = scorers_nms
        self.__flush()
    
    def __flush(self):
        self.cache = {key:None for key in self.scorers_nms}
        
    def update_cache(self, res):
        self.cache = res
    
    def eval(self, algo: AlgoProtocol, epoch, total_step)->Dict:
        raise NotImplementedError
    
    def __call__(self, algo: AlgoProtocol, epoch, total_step):
        self.__flush()
        res = self.eval(algo, epoch, total_step)
        self.update_cache(res)
                
    def __getitem__(self, i):
        return self.cache[i]
    
    def __len__(self):
        return len(self.scorers_nms)

class WrapperAccessor:
    """ Helper class for accessing results stored in the cache of a 'Wrapper'
    object
    """
    def __init__(self, item_key, wrapper:Wrapper) -> None:
        self.__wrapper = wrapper
        self.__item_key = item_key
        
    def __call__(self, *args, **kwargs):
        return self.__wrapper[self.__item_key]
    
    
class EpochCallbackHandler:
    """Helper class for executing multiple wrapper objectes with a single calls
    """
    def __init__(self, wrappers:List[Wrapper]) -> None:
        self.__wrappers = wrappers
        
    def __call__(self, algo, epoch, total_step) -> Any:
        for i in self.__wrappers:
            i(algo, epoch, total_step)