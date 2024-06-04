from typing import Any, Sequence, Union, Protocol
import numpy.typing as npt
from d3rlpy.models.torch.policies import ActionOutput

NDArray = npt.NDArray[Any]
Observation = Union[NDArray, Sequence[NDArray]]


class D3rlpyAlgoPredictProtocal(Protocol):
    
    def __call__(self, x:Observation) -> NDArray:
        ... 
        
class D3rlpyPolicyProtocal(Protocol):
    
    def __call__(self, x:Observation) -> ActionOutput:
        ... 