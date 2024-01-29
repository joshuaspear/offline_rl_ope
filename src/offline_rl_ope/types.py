import torch
from typing import Any, Dict, Union
import numpy.typing as npt
import numpy as np
from sklearn.base import BaseEstimator

__all__ = [
    "NDArray", 
    "Float32NDArray",
    "PropensityTorchOutputType"
    ]

NDArray = npt.NDArray[Any]
Float32NDArray = npt.NDArray[np.float32]
PropensityTorchOutputType = Dict[str,torch.Tensor]
