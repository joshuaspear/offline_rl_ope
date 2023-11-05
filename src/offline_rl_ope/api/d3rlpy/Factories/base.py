from abc import ABCMeta, abstractmethod
from typing import Any, Dict
from d3rlpy.metrics import EvaluatorProtocol

class EvaluatorFactoryBase(metaclass=ABCMeta):
        
    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Dict[str,EvaluatorProtocol]:
        pass