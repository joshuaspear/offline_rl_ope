from typing import Any, List
import numpy as np
import torch

test_state_vals = [
    [[1,2,3,4], [5,6,7,8], [5,7,2,9], [5,7,2,9]],
    [[5,6,7,8], [5,6,7,8], [1,2,3,4]]
]
test_action_vals = [
    [[1], [0], [0], [1]],
    [[0], [0], [1]]
]

test_action_probs = [
    [[0.9], [0.7], [0.66], [0.7]],
    [[0.54], [0.9], [0.5]]
]

test_eval_action_vals = [
    [[1], [1], [0], [1]],
    [[0], [0], [0]]
]

test_eval_action_probs = [
    [[1], [0.07], [0.89], [1]],
    [[0.75], [0.9], [0.2]]
]

test_reward_values = [
    [[1],[-1], [1], [1]],
    [[-1],[-1], [-1]]
]

test_act_indiv_weights = [
    np.array([1/0.9, 0.07/0.7, 0.89/0.66, 1/0.7]),
    np.array([ 0.75/0.54, 0.9/0.9, 0.2/0.5])
    ]

weight_test_res = torch.Tensor(
    [
        test_act_indiv_weights[0].tolist(),
        [*test_act_indiv_weights[1].tolist(),0]
        ]
)
msk_test_res = torch.Tensor(
    [
        [1]*4,
        [*[1]*3,0]
    ]
)

reward_test_res = torch.Tensor(
    [
        test_reward_values[0],
        [*test_reward_values[1],[0]]
        ]
).squeeze()


def flatten_lst(input_lst:List[Any], recursive:bool=True)->List[Any]:
    """Function for flattening a list containing lists

    Args:
        input_lst (List[Any]): Input list to flatten
        recursive (bool, optional): If true the function will recursively 
        flatten lists within the input list else only the first layer of lists
        will be flattened. Defaults to True.

    Returns:
        List[Any]: A flattened version of the input_lst
    """
    output_lst = []
    for sub_lst in input_lst:
        if isinstance(sub_lst, list):
            if recursive:
                sub_lst = flatten_lst(sub_lst)
            output_lst = output_lst + sub_lst
        else:
            output_lst.append(sub_lst)
    return output_lst