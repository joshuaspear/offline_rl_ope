from abc import ABCMeta, abstractmethod
from typing import Any, Dict
from d3rlpy.interface import QLearningAlgoProtocol

class OPECallbackBase(metaclass=ABCMeta):
    
    def __init__(self, debug_path) -> None:
        self.debug_path = debug_path
        if self.debug_path is not None:
            self._debug = self.debug_true
        else:
            self._debug = lambda algo, epoch, total_step: None
    
    @abstractmethod
    def debug_true(
        self, 
        algo:QLearningAlgoProtocol,  
        epoch:int, 
        total_step:int
        ) -> None:
        pass
    
    @abstractmethod
    def run(
        self,
        algo:QLearningAlgoProtocol,  
        epoch:int, 
        total_step:int
        ) -> None:
        pass
    
    def __call__(
        self, 
        algo:QLearningAlgoProtocol,  
        epoch:int, 
        total_step:int
        ) -> None:
        self.run(algo, epoch, total_step)
        self._debug(algo, epoch, total_step)


class QueryCallbackBase(OPECallbackBase):
    
    def __init__(self, debug_path) -> None:
        super().__init__(debug_path=debug_path)
        self.cache:Dict[str, Any] = {}

    def __getitem__(self, idx:str):
        return self.cache[idx]
