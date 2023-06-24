from abc import ABCMeta, abstractmethod
from d3rlpy.algos.base import AlgoBase
from typing import Callable
import torch

class DirectMethodBase(metaclass=ABCMeta):
    
    def __init__(self, model: Callable) -> None:
        self.model = model
    
    @abstractmethod
    def get_v(self, state:torch.Tensor):
        pass
    
    @abstractmethod
    def get_q(self, state:torch.Tensor, action:torch.Tensor):
        pass
    
    
class D3rlpyQlearnDM(DirectMethodBase):
    
    def __init__(self, model:AlgoBase) -> None:
        super().__init__(model=model)
    
    def get_q(self, state:torch.Tensor, action:torch.Tensor):
        values = torch.Tensor(self.model.predict_value(x=state, action=action))
        return values
        
    def get_v(self, state:torch.Tensor):
        actions = self.model.predict(state)
        values = torch.Tensor(self.model.predict_value(state, actions))
        return values