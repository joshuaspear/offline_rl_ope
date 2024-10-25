from abc import abstractmethod
import torch
import torch.nn as nn
from typing import List

from ....types import PropensityTorchOutputType, StateTensor

__all__ = [
    "PropensityTorchBase",
    "PropensityTorchMlpBase"
    ]

class PropensityTorchBase(nn.Module):
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x:torch.Tensor) -> PropensityTorchOutputType:
        pass
        

class PropensityTorchMlpBase(PropensityTorchBase):
    
    def __init__(
        self, 
        input_dim:int, 
        layers_dim:List[int], 
        actvton:nn.Module, 
        init_bias:float = 0
        ) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        if len(layers_dim) > 0:
            # Add the hidden layer mapping the input
            self.layers.append(
                nn.Linear(
                    in_features=input_dim, out_features=layers_dim[0])
                )
            # Add the remaining hidden layers
            for i in range(len(layers_dim)-1):
                self.layers.append(actvton)
                self.layers.append(nn.Linear(
                    in_features=layers_dim[i], 
                    out_features=layers_dim[i+1]
                ))
        
        self.__init_weights(init_bias)
    
    def __init_weights(self, value):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(value)            
    
    @abstractmethod
    def forward(self, x:StateTensor) -> PropensityTorchOutputType:
        pass
