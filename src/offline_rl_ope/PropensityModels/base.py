from abc import ABCMeta, abstractmethod

__all__ = [
    "PropensityTrainer"
]

class PropensityTrainer(metaclass=ABCMeta):
            
    @abstractmethod
    def save(self, path:str):
        pass