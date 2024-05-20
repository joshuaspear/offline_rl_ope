from .types import ArrayType
from typing import List


def check_array_dim(
    array_type:ArrayType, 
    dim:int
    )->None:
    assert len(array_type.shape) == dim, f"Expected array dimension: {dim}"

def check_array_shape(
    array_type:ArrayType, 
    arr_shape:List[int]
    )->None:
    assert array_type.shape == arr_shape, f"Expected array shape: {arr_shape}"