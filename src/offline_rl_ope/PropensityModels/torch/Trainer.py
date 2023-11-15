import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import pickle

from ..base import PropensityTrainer
from .models.base import PropensityTorchBase
from ...types import Float32NDArray

__all__ = [
    "TorchPropensityTrainer",
    "TorchClassTrainer",
    "TorchRegTrainer"
]

class TorchPropensityTrainer(PropensityTrainer):
    
    def __init__(
        self, 
        estimator:PropensityTorchBase, 
        gpu:bool,
        ) -> None:
        super().__init__(estimator=estimator)
        self.gpu = gpu
        if self.gpu:
            self.input_setup = self.load_to_gpu
        else:
            self.input_setup = self.load_pass
    
    def load_to_gpu(self, x:torch.Tensor):
        x = x.to(device='cuda')
        return x
    
    def load_pass(self, x:torch.Tensor):
        return x
    
    def to_cpu(self):
        self.input_setup = self.load_pass
        self.estimator = self.estimator.to(device='cpu')
    
    def to_gpu(self):
        self.input_setup = self.load_to_gpu
        self.estimator = self.estimator.to(device='cuda')
        
    def save(self, path:str) -> None:
        if self.gpu:
            self.to_cpu()
        with open(path, "wb") as f:
            pickle.dump(self, f)
        if self.gpu:
            self.to_gpu()
               
class TorchClassTrainer(TorchPropensityTrainer):
    def __init__(
        self, 
        estimator:nn.Module, 
        gpu:bool
        ) -> None:
        
        super().__init__(estimator=estimator, gpu=gpu)

    def predict(
        self, 
        x:Float32NDArray, 
        *args, 
        **kwargs
        ) -> Float32NDArray:
        x = torch.Tensor(x)
        x = self.input_setup(x)
        self.estimator.eval()
        res = self.estimator(x)
        # Take max over values
        res = torch.argmax(res["out"], dim=1, keepdim=False)
        res = res.cpu().detach().numpy()
        return res
    
    def predict_proba(
        self, 
        x: Float32NDArray, 
        y: Float32NDArray, 
        *args, 
        **kwargs
        ) -> Float32NDArray:
        x = torch.Tensor(x)
        x = self.input_setup(x)
        self.estimator.eval()
        res = self.estimator(x)
        res_out = res["out"].cpu().detach().numpy()
        n_rows = res_out.shape[0]
        n_out = res_out.shape[2]
        dim_0_sub = np.arange(0,n_rows)[:,None]
        dim_1_sub = np.tile(np.arange(0,n_out), (n_rows,1))
        res_out = res_out[dim_0_sub,y.astype(int),dim_1_sub]
        return res_out
        

class TorchRegTrainer(TorchPropensityTrainer):
    
    def __init__(
        self, 
        estimator:nn.Module, 
        dist_func:torch.distributions.Distribution, 
        gpu:bool
        ) -> None:
        super().__init__(estimator=estimator, gpu=gpu)
        self.dist_func = dist_func
              
    def predict(
        self, 
        x:Float32NDArray, 
        *args, 
        **kwargs
        ) -> Float32NDArray:
        x = torch.Tensor(x)
        x = self.input_setup(x)
        self.estimator.eval()
        res = self.estimator(x)
        res = res["loc"].cpu().detach().numpy()
        return res
    
    def predict_proba(
        self, 
        x: Float32NDArray, 
        y: Float32NDArray, 
        *args, 
        **kwargs
        ) -> Float32NDArray:
        x = torch.Tensor(x)
        x = self.input_setup(x)
        y = torch.Tensor(y)
        y = self.input_setup(y)
        self.estimator.eval()
        pred_res = self.estimator(x)
        d_f = self.dist_func(**pred_res)
        res = torch.exp(d_f.log_prob(y)).cpu().detach().numpy()
        return res