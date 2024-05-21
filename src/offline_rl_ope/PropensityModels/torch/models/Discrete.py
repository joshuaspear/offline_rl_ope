from typing import List
import torch
import torch.nn as nn

from .base import PropensityTorchMlpBase
from ....types import PropensityTorchOutputType, StateTensor

__all__ = [
    "Categorical"
]

class Categorical(PropensityTorchMlpBase):
    
    def __init__(
        self, 
        input_dim:int, 
        layers_dim:List[int], 
        out_dim:List[int], 
        actvton=nn.ReLU(),
        init_bias:float=0, 
        ) -> None:
        
        super().__init__(input_dim=input_dim, layers_dim=layers_dim, 
                         actvton=actvton, init_bias=init_bias)
        
        self.layers.append(actvton)
        # Add the final layer
        self.out_layers = nn.ModuleList()
        for head_dim in out_dim:
            self.out_layers.append(nn.Linear(
                in_features=layers_dim[-1],
                out_features=head_dim
            ))
        self.out_actvton = nn.Sigmoid()
        
    def forward(
        self, 
        x:StateTensor
        ) -> PropensityTorchOutputType:
        for layer in self.layers:
            x = layer(x)
        out:List[torch.Tensor] = []
        for head in self.out_layers:
            out_val = head(x)
            out.append(self.out_actvton(out_val)[:,:,None])
        res = torch.concat(out, dim=2)
        return {"out": res}
