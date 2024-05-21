from abc import ABCMeta, abstractmethod
from d3rlpy.algos import QLearningAlgoBase
from typing import Callable
import torch
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker

from ..types import StateTensor, ActionTensor
from ..RuntimeChecks import check_array_dim

class DirectMethodBase(metaclass=ABCMeta):
    
    def __init__(self, model: Callable) -> None:
        self.model = model
    
    @abstractmethod
    def calculate_v(self, state:torch.Tensor) -> torch.Tensor:
        pass
    
    @abstractmethod
    def calculate_q(self, state:torch.Tensor, action:torch.Tensor) -> torch.Tensor:
        pass

    @jaxtyped(typechecker=typechecker)
    def get_v(self, state:torch.Tensor) -> torch.Tensor:
        res = self.calculate_v(state=state)
        check_array_dim(res,2)
        return res 
    
    @jaxtyped(typechecker=typechecker)
    def get_q(
        self, 
        state:StateTensor, 
        action:ActionTensor
        ) -> Float[torch.Tensor, "traj_length 1"]:
        res = self.calculate_q(state=state, action=action)
        check_array_dim(res,2)
        return res 
    
    
class D3rlpyQlearnDM(DirectMethodBase):
    
    def __init__(
        self, 
        model:QLearningAlgoBase
        ) -> None:
        assert isinstance(model,QLearningAlgoBase)
        super().__init__(model=model)
    
    @jaxtyped(typechecker=typechecker)
    def calculate_q(
        self, 
        state:StateTensor, 
        action:ActionTensor
        ) -> Float[torch.Tensor, "traj_length 1"]:
        # assert isinstance(state,torch.Tensor)
        # assert isinstance(action,torch.Tensor) 
        values = torch.tensor(self.model.predict_value(
            x=state.cpu().numpy(), action=action.cpu().numpy())).reshape(-1,1)
        return values
    
    @jaxtyped(typechecker=typechecker)
    def calculate_v(
        self, 
        state:StateTensor
        ) -> Float[torch.Tensor, "traj_length 1"]:
        # assert isinstance(state,torch.Tensor)
        state = state.numpy()
        actions = self.model.predict(state)
        values = torch.tensor(self.model.predict_value(
            x=state, action=actions)).reshape(-1,1)
        return values