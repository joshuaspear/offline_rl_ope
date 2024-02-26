from typing import Any, Sequence, Union, Protocol
import numpy.typing as npt

NDArray = npt.NDArray[Any]
Observation = Union[NDArray, Sequence[NDArray]]


class D3rlpyAlgoPredictProtocal(Protocol):
    
    def __call__(self, x:Observation) -> NDArray:
        ... 