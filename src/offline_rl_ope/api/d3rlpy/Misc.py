from typing import Callable
import torch

__all__ = ["D3RlPyTorchAlgoPredict"]

class D3RlPyTorchAlgoPredict:
    
    def __init__(self, predict_func:Callable):
        self.predict_func = predict_func
        
    def __call__(self, x:torch.Tensor):
        pred = self.predict_func(x.cpu().numpy())
        return torch.Tensor(pred)
