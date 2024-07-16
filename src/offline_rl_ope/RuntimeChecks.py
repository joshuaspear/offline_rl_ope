from .types import ArrayType
from typing import List


def check_array_dim(
    array_type:ArrayType, 
    dim:int
    )->None:
    test_val = len(array_type.shape)
    _msg = f"Expected array dimension: {dim}. Found: {test_val}"
    assert test_val == dim, _msg

def check_array_shape(
    array_type:ArrayType, 
    arr_shape:List[int]
    )->None:
    test_val = list(array_type.shape)
    _msg = f"Expected array dimension: {arr_shape}. Found: {test_val}"
    assert test_val == arr_shape, _msg