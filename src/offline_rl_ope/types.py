import torch
from typing import Any, Dict, Union, Protocol, runtime_checkable
import numpy.typing as npt
import numpy as np
import pandas as pd

__all__ = [
    "NDArray", 
    "Float32NDArray",
    "PropensityTorchOutputType"
    ]

NDArray = npt.NDArray[Any]
Float32NDArray = npt.NDArray[np.float32]
PropensityTorchOutputType = Dict[str,torch.Tensor]
ArrayType = Union[pd.DataFrame,np.ndarray,torch.Tensor]

@runtime_checkable
class PropensityTorchBaseType(Protocol):
    
    def __call__(self, x:torch.Tensor) -> PropensityTorchOutputType:
        ...
    
    def eval(self)->None:
        ...