import torch
from .types import D3rlpyAlgoPredictProtocal

from ...RuntimeChecks import check_array_dim

__all__ = ["D3RlPyTorchAlgoPredict"]

class D3RlPyTorchAlgoPredict:
    
    def __init__(self, predict_func:D3rlpyAlgoPredictProtocal):
        self.predict_func = predict_func
        
    def __call__(self, x:torch.Tensor):
        pred = self.predict_func(x.cpu().numpy())
        check_array_dim(pred,1)
        pred = pred.reshape(-1,1)
        return torch.Tensor(pred)
