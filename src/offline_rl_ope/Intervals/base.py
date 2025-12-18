from abc import ABCMeta, abstractmethod
from typing import Tuple


class Interval(metaclass=ABCMeta):
    
    @abstractmethod
    def predict(*args, **kwargs)->Tuple[float,float]:
        pass