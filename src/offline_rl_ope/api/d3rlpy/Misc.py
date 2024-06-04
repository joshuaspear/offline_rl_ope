import torch
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker
from d3rlpy.models.torch.policies import build_squashed_gaussian_distribution

from .types import (
    D3rlpyAlgoPredictProtocal, D3rlpyPolicyProtocal,
    )
from ...types import StateTensor, ActionTensor, TorchPolicyReturn



__all__ = [
    "D3RlPyDeterministicWrapper", "D3RlPyDeterministicDiscreteWrapper",
    "D3RlPyStochasticWrapper"
    ]

class D3RlPyDeterministicWrapper:

    def __init__(
        self, 
        predict_func:D3rlpyAlgoPredictProtocal,
        action_dim:int
        ):
        self.predict_func = predict_func
        self.action_dim = action_dim
    
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
        
class D3RlPyStochasticWrapper:
    
    def __init__(
        self, 
        policy_func:D3rlpyPolicyProtocal,
        ) -> None:
        self.policy_func = policy_func
    
    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        state: StateTensor, 
        action: ActionTensor
        ) -> TorchPolicyReturn:
        dist = build_squashed_gaussian_distribution(self.policy_func(state))
        with torch.no_grad():
            res = torch.exp(dist.log_prob(action))
        return TorchPolicyReturn(
            actions=action,
            action_prs=res
            )
        
    