from abc import ABCMeta, abstractmethod
from d3rlpy.algos import QLearningAlgoBase
from typing import Callable
import torch

class DirectMethodBase(metaclass=ABCMeta):
    
    def __init__(self, model: Callable) -> None:
        self.model = model
    
    @abstractmethod
    def get_v(self, state:torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def get_q(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        pass
    
    
class D3rlpyQlearnDM(DirectMethodBase):
    
    def __init__(self, model:QLearningAlgoBase) -> None:
        super().__init__(model=model)
    
    def get_q(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        values = torch.tensor(self.model.predict_value(
            x=state.numpy(), action=action.numpy()))
        return values
        
    def get_v(self, state:torch.Tensor) -> torch.Tensor:
        state = state.numpy()
        actions = self.model.predict(state)
        values = torch.tensor(self.model.predict_value(
            x=state, action=actions))
        return values