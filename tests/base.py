from dataclasses import dataclass
from typing import Any, List, Dict
import numpy as np
import torch
import copy

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
    traj_is_weights_is:torch.Tensor = None
    traj_is_weights_pd:torch.Tensor = None
    weight_test_res_alter:torch.Tensor = None
    traj_is_weights_is_alter:torch.Tensor = None
    traj_is_weights_pd_alter:torch.Tensor = None
    msk_test_res:torch.Tensor = None
    reward_test_res:torch.Tensor = None
    
    @staticmethod
    def __get_traj_weights(
        weight_test_res:torch.Tensor, 
        msk_test_res:torch.Tensor
        ):
        # Taking product to define \prod_{t=0}^{H}w_{t,i}
        _traj_is_weights_sub = weight_test_res.detach().clone()
        _traj_is_weights_sub[msk_test_res == 0] = 1
        _traj_is_weights_is = _traj_is_weights_sub.prod(dim=1, keepdim=True)
        traj_is_weights_is = _traj_is_weights_is.repeat(
            (1,weight_test_res.shape[1])
            )
        traj_is_weights_pd = _traj_is_weights_sub.cumprod(
            dim=1)
        traj_is_weights_is[msk_test_res == 0] = 0
        traj_is_weights_pd[msk_test_res == 0] = 0
        return traj_is_weights_is, traj_is_weights_pd
    
    @staticmethod
    def __get_weight_mask_matrix(
        test_act_indiv_weights:List[np.array]
        ):
        max_len = max([len(i) for i in test_act_indiv_weights])
        weight_test_res = []
        msk_test_res = []
        for i in test_act_indiv_weights:
            weight_test_res.append(np.pad(i,(0,max_len-len(i))).tolist())
            msk_test_res.append(
                np.pad(i.astype(bool),(0,max_len-len(i))).tolist()
                )
        return torch.Tensor(weight_test_res), torch.Tensor(msk_test_res).float()
    
    def __post_init__(self):
        test_act_indiv_weights = []
        for i,j in zip(self.test_eval_action_probs,self.test_action_probs):
            test_act_indiv_weights.append(
                np.array(i).squeeze()/np.array(j).squeeze()
                )
        self.test_act_indiv_weights = test_act_indiv_weights
        
        (
            self.weight_test_res, 
            self.msk_test_res
            ) = self.__get_weight_mask_matrix(
                self.test_act_indiv_weights
            )
        
        (
            self.traj_is_weights_is, 
            self.traj_is_weights_pd
            ) = self.__get_traj_weights(
                self.weight_test_res, self.msk_test_res
                )
        # Check for trivial weights i.e., all 0
        assert not (self.traj_is_weights_is == 0).all().item(), "Weights are trivial"
        assert not (self.traj_is_weights_pd == 0).all().item(), "Weights are trivial" 
        
        self.test_act_indiv_weights_alter = copy.deepcopy(
            self.test_act_indiv_weights)
        self.test_act_indiv_weights_alter[0] = np.zeros(
            len(self.test_act_indiv_weights_alter[0])
        )
        
        self.weight_test_res_alter,_ = self.__get_weight_mask_matrix(
            self.test_act_indiv_weights_alter
        )
        
        (self.traj_is_weights_is_alter, 
         self.traj_is_weights_pd_alter) = self.__get_traj_weights(
            self.weight_test_res_alter, self.msk_test_res
        )
         
        max_len = max([len(i) for i in test_act_indiv_weights])
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

test_configs:Dict[str,TestConfig] = {}

test_configs.update(
    {
        "binary_action": TestConfig(
            test_state_vals=test_state_vals,
            test_action_vals=test_action_vals,
            test_action_probs=test_action_probs,
            test_eval_action_vals=test_eval_action_vals,
            test_eval_action_probs=test_eval_action_probs,
            test_reward_values=test_reward_values,
            test_dm_s_values=test_dm_s_values,
            test_dm_sa_values=test_dm_sa_values
            )
        }
)

test_action_vals = [
    [[1], [0], [2], [2]],
    [[0], [2], [1]]
]

test_eval_action_vals = [
    [[1], [1], [2], [2]],
    [[0], [0], [1]]
]


test_configs.update(
    {
        "categorical_action": TestConfig(
            test_state_vals=test_state_vals,
            test_action_vals=test_action_vals,
            test_action_probs=test_action_probs,
            test_eval_action_vals=test_eval_action_vals,
            test_eval_action_probs=test_eval_action_probs,
            test_reward_values=test_reward_values,
            test_dm_s_values=test_dm_s_values,
            test_dm_sa_values=test_dm_sa_values
            )
        }
)

test_action_vals = [
    [[1,1], [0,1], [0,1], [1,0]],
    [[0,1], [0,0], [1,0]]
]

test_eval_action_vals = [
    [[1,1], [1,1], [0,1], [1,0]],
    [[0,1], [0,0], [0,1]]
]

test_configs.update(
    {
        "multi_binary_action": TestConfig(
            test_state_vals=test_state_vals,
            test_action_vals=test_action_vals,
            test_action_probs=test_action_probs,
            test_eval_action_vals=test_eval_action_vals,
            test_eval_action_probs=test_eval_action_probs,
            test_reward_values=test_reward_values,
            test_dm_s_values=test_dm_s_values,
            test_dm_sa_values=test_dm_sa_values
            )
        }
)

test_configs_fmt = [[key,test_configs[key]] for key in test_configs.keys()]
test_configs_fmt_class = [
    {"test_conf":test_configs[key]} for key in test_configs.keys()
    ]

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

tmp = test_configs["binary_action"]
