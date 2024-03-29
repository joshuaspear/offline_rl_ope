import unittest
import torch
import logging
import numpy as np
import copy
from offline_rl_ope.Metrics import ValidWeightsProp
from ..base import weight_test_res, msk_test_res

logger = logging.getLogger("offline_rl_ope")


class TestImportanceCalc:
    
    def __init__(self) -> None:
        self.weight_msk = msk_test_res

class TestImportanceSampler:
    
    def __init__(self) -> None:
        self.is_weight_calc = None
        self.traj_is_weights = weight_test_res
        self.is_weight_calc = TestImportanceCalc()

class TestValidWeightsProp(unittest.TestCase):

    def test_call(self):
        max_val=10000
        min_val=0.000001
        num = (weight_test_res > min_val) & (weight_test_res < max_val)
        num = torch.sum(num, axis=1)
        denum = torch.sum(msk_test_res, axis=1)
        act_res = torch.mean(num/denum).item()
        metric = ValidWeightsProp(
            is_obj=TestImportanceSampler(), 
            max_w=max_val,
            min_w=min_val
            )
        pred_res = metric()
        self.assertEqual(act_res,pred_res)