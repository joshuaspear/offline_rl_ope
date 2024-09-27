from abc import abstractmethod
from typing import Optional, Union
import torch
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker
from d3rlpy.models.torch.policies import build_squashed_gaussian_distribution
from d3rlpy.preprocessing import ActionScaler, ObservationScaler
from .types import (
    D3rlpyAlgoPredictProtocal, D3rlpyPolicyProtocal,
    )
from ...types import StateTensor, ActionTensor, TorchPolicyReturn



__all__ = [
    "D3RlPyDeterministicWrapper", "D3RlPyDeterministicDiscreteWrapper",
    "D3RlPyStochasticWrapper"
    ]

class D3RlPyWrapperBase:
    def __init__(
        self, 
        predict_func:Union[D3rlpyAlgoPredictProtocal,D3rlpyPolicyProtocal]
        ):
        self.predict_func = predict_func
    
    @abstractmethod
    def __call__(self, x:StateTensor)->TorchPolicyReturn:
        pass


class D3RlPyDeterministicWrapper(D3RlPyWrapperBase):

    def __init__(
        self, 
        predict_func: Union[D3rlpyAlgoPredictProtocal, D3rlpyPolicyProtocal], 
        action_dim: int
        ):
        self.action_dim = action_dim
        super().__init__(
            predict_func=predict_func
            )
        
    @jaxtyped(typechecker=typechecker)
    def __call__(self, x:StateTensor)->TorchPolicyReturn:
        pred = self.predict_func(x.cpu().numpy()).reshape(
            -1, self.action_dim
            )
        return TorchPolicyReturn(
            actions=torch.Tensor(pred),
            action_prs=None
            )

class D3RlPyDeterministicDiscreteWrapper(D3RlPyDeterministicWrapper):
    
    def __init__(
        self, 
        predict_func:D3rlpyAlgoPredictProtocal,
        action_dim:int
        ):
        assert action_dim==1, "D3RlPy action dimension is 1 for discrete tasks"
        super().__init__(
            predict_func=predict_func,
            action_dim=action_dim
            )


class D3RlPyStochasticWrapper(D3RlPyWrapperBase):
    
    def __init__(
        self, 
        policy_func:D3rlpyPolicyProtocal,
        observation_scaler:Optional[ObservationScaler] = None,
        action_scaler:Optional[ActionScaler]=None
        ) -> None:
        super().__init__(
            predict_func=policy_func
            )
        if action_scaler is not None:
            assert action_scaler.built, "Action scaler is not built"
        self.action_scaler = action_scaler
        if observation_scaler is not None:
            assert observation_scaler.built, "Observation scaler is not built"
        self.observation_scaler = observation_scaler

        
    
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        state: StateTensor, 
        action: ActionTensor
        ) -> TorchPolicyReturn:
        if self.observation_scaler is not None:
            x_scaled = self.observation_scaler.transform(x=state)
        else:
            x_scaled = state
        dist = build_squashed_gaussian_distribution(
            self.predict_func(x_scaled)
            )
        if self.action_scaler is not None:
            scaled_action = self.action_scaler.transform(x=action)
        else:
            scaled_action = action
        with torch.no_grad():
            res = torch.exp(dist.log_prob(scaled_action))
        return TorchPolicyReturn(
            actions=action,
            action_prs=res
            )
        
    