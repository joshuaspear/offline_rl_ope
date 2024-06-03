import torch
from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from .types import D3rlpyAlgoPredictProtocal
from ...types import StateTensor, ActionTensor



__all__ = ["D3RlPyTorchAlgoPredict"]

class D3RlPyTorchAlgoPredict:
    
    def __init__(
        self, 
        predict_func:D3rlpyAlgoPredictProtocal,
        action_dim:int
        ):
        self.predict_func = predict_func
        self.action_dim = action_dim
    
    @jaxtyped(typechecker=typechecker)
    def __call__(self, x:StateTensor)->ActionTensor:
        pred = self.predict_func(x.cpu().numpy()).reshape(
            -1, self.action_dim
            )
        return torch.Tensor(pred)
