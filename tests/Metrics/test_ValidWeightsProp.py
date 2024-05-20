import unittest
import torch
import numpy as np
import copy
from offline_rl_ope.Metrics import ValidWeightsProp
from offline_rl_ope import logger
# from ..base import weight_test_res, msk_test_res
from ..base import (
    single_discrete_action_test as sdat,
    duel_discrete_action_test as ddat,
    bin_discrete_action_test as bdat
    )

for test_conf in [sdat,ddat,bdat]:
    class TestValidWeightsProp(unittest.TestCase):

        def test_call(self):
            max_val=10000
            min_val=0.000001
            num = (test_conf.weight_test_res > min_val) & (test_conf.weight_test_res < max_val)
            num = torch.sum(num, axis=1)
            denum = torch.sum(test_conf.msk_test_res, axis=1)
            act_res = torch.mean(num/denum).item()
            metric = ValidWeightsProp(
                max_w=max_val,
                min_w=min_val
                )
            pred_res = metric(weights=test_conf.weight_test_res, weight_msk=test_conf.msk_test_res)
            self.assertEqual(act_res,pred_res)