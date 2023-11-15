import torch
from typing import Any, Dict, Union
import numpy.typing as npt
import numpy as np
from sklearn.base import BaseEstimator

__all__ = [
    "PropensityEstimatorType",
    "PropensityTorchOutputType"
    ]

NDArray = npt.NDArray[Any]
Float32NDArray = npt.NDArray[np.float32]
PropensityEstimatorType = Union[torch.nn.Module, BaseEstimator]
PropensityTorchOutputType = Dict[str,torch.Tensor]
