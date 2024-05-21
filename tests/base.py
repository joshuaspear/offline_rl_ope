from dataclasses import dataclass
from typing import Any, List
import numpy as np
import torch

@dataclass
class TestConfig:
    
    test_state_vals:List[List[int]]
    test_action_vals:List[List[int]]
    test_action_probs:List[List[float]]
    test_eval_action_vals:List[List[int]]
    test_eval_action_probs:List[List[int]]
    test_reward_values:List[List[float]]
    test_dm_s_values:List[List[float]]
    test_dm_sa_values:List[List[float]]
    test_act_indiv_weights:List[np.ndarray[float]] = None
    weight_test_res:torch.Tensor = None
    msk_test_res:torch.Tensor = None
    reward_test_res:torch.Tensor = None
    
    def __post_init__(self):
        test_act_indiv_weights = []
        for i,j in zip(self.test_eval_action_probs,self.test_action_probs):
            test_act_indiv_weights.append(
                np.array(i).squeeze()/np.array(j).squeeze()
                )
        self.test_act_indiv_weights = test_act_indiv_weights
        
        max_len = max([len(i) for i in self.test_act_indiv_weights])
        weight_test_res = []
        msk_test_res = []
        for i in self.test_act_indiv_weights:
            weight_test_res.append(np.pad(i,(0,max_len-len(i))).tolist())
            msk_test_res.append(
                np.pad(i.astype(bool),(0,max_len-len(i))).tolist()
                )
        self.weight_test_res = torch.Tensor(weight_test_res)
        self.msk_test_res = torch.Tensor(msk_test_res).float()
        
        reward_test_res = []
        for i in self.test_reward_values:
            reward_test_res.append(
                np.pad(np.array(i).squeeze(),(0,max_len-len(i))).tolist()
                )
        self.reward_test_res = torch.Tensor(reward_test_res).float()
    

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

test_dm_s_values = [
    [[0.8],[-1], [0.5], [0.4]],
    [[-2],[-1], [-0.5]]
]

test_dm_sa_values = [
    [[0.7],[-3], [0.5], [0.6]],
    [[-3],[-2], [-0.8]]
]

# test_act_indiv_weights = [
#     np.array([1/0.9, 0.07/0.7, 0.89/0.66, 1/0.7]),
#     np.array([ 0.75/0.54, 0.9/0.9, 0.2/0.5])
#     ]


# weight_test_res = torch.Tensor(
#     [
#         test_act_indiv_weights[0].tolist(),
#         [*test_act_indiv_weights[1].tolist(),0]
#         ]
# )

# msk_test_res = torch.Tensor(
#     [
#         [1]*4,
#         [*[1]*3,0]
#     ]
# )

# reward_test_res = torch.Tensor(
#     [
#         test_reward_values[0],
#         [*test_reward_values[1],[0]]
#         ]
# ).squeeze()


bin_discrete_action_test = TestConfig(
    test_state_vals=test_state_vals,
    test_action_vals=test_action_vals,
    test_action_probs=test_action_probs,
    test_eval_action_vals=test_eval_action_vals,
    test_eval_action_probs=test_eval_action_probs,
    test_reward_values=test_reward_values,
    test_dm_s_values=test_dm_s_values,
    test_dm_sa_values=test_dm_sa_values
)

test_action_vals = [
    [[1], [0], [2], [2]],
    [[0], [2], [1]]
]

test_eval_action_vals = [
    [[1], [1], [2], [2]],
    [[0], [0], [1]]
]

single_discrete_action_test = TestConfig(
    test_state_vals=test_state_vals,
    test_action_vals=test_action_vals,
    test_action_probs=test_action_probs,
    test_eval_action_vals=test_eval_action_vals,
    test_eval_action_probs=test_eval_action_probs,
    test_reward_values=test_reward_values,
    test_dm_s_values=test_dm_s_values,
    test_dm_sa_values=test_dm_sa_values
)


test_action_vals = [
    [[1,1], [0,1], [0,1], [1,0]],
    [[0,1], [0,0], [1,0]]
]

test_eval_action_vals = [
    [[1,1], [1,1], [0,1], [1,0]],
    [[0,1], [0,0], [0,1]]
]

duel_discrete_action_test = TestConfig(
    test_state_vals=test_state_vals,
    test_action_vals=test_action_vals,
    test_action_probs=test_action_probs,
    test_eval_action_vals=test_eval_action_vals,
    test_eval_action_probs=test_eval_action_probs,
    test_reward_values=test_reward_values,
    test_dm_s_values=test_dm_s_values,
    test_dm_sa_values=test_dm_sa_values
)

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