import torch
from typing import Any, Dict, Union, Protocol
import numpy.typing as npt
import numpy as np

__all__ = [
    "NDArray", 
    "Float32NDArray",
    "PropensityTorchOutputType"
    ]

NDArray = npt.NDArray[Any]
Float32NDArray = npt.NDArray[np.float32]
PropensityTorchOutputType = Dict[str,torch.Tensor]


class PropensityTorchBaseType(Protocol):
    
    def __call__(self, x:torch.Tensor) -> PropensityTorchOutputType:
        ...
    
    def eval(self)->None:
        ...