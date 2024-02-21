from typing import List
import torch
import torch.nn as nn

from .base import PropensityTorchMlpBase
from ....types import PropensityTorchOutputType

__all__ = [
    "FullGuassian"
    ]

class FullGuassian(PropensityTorchMlpBase):
    
    def __init__(self, input_dim:int,  layers_dim:List[int], m_out_dim:int,
                 sd_out_dim: int, actvton=nn.ReLU(), init_bias:float=0
                ) -> None:
        
        super().__init__(input_dim=input_dim, layers_dim=layers_dim, 
                         actvton=actvton, init_bias=init_bias)
        # For homoscedastic variance made sd_out_dim = 1
        if m_out_dim != sd_out_dim:
            if sd_out_dim != 1:
                raise Exception
        self.layers.append(actvton)
        # Add the independent mean layer
        self.m_layer = nn.Linear(
            in_features=layers_dim[-1],
            out_features=m_out_dim
        )
        # Add the independent sd layer
        self.sd_layer = nn.Linear(
            in_features=layers_dim[-1],
            out_features=sd_out_dim
        )
        
    def forward(self, x) -> PropensityTorchOutputType:
        for layer in self.layers:
            x = layer(x)
        m_out = self.m_layer(x)
        sd_out = torch.exp(self.sd_layer(x))
        return {"loc": m_out, "scale": sd_out}