import unittest
import torch
import numpy as np
import copy
from offline_rl_ope.Metrics import ValidWeightsProp
from offline_rl_ope import logger
from parameterized import parameterized_class
from ..base import test_configs_fmt_class, TestConfig

@parameterized_class(test_configs_fmt_class)
class TestValidWeightsProp(unittest.TestCase):

    test_conf:TestConfig
    
    def test_call(self):
        max_val=10000
        min_val=0.000001
        num = (self.test_conf.weight_test_res > min_val) & (self.test_conf.weight_test_res < max_val)
        num = torch.sum(num, axis=1)
        denum = torch.sum(self.test_conf.msk_test_res, axis=1)
        act_res = torch.mean(num/denum).item()
        metric = ValidWeightsProp(
            max_w=max_val,
            min_w=min_val
            )
        pred_res = metric(weights=self.test_conf.weight_test_res, weight_msk=self.test_conf.msk_test_res)
        self.assertEqual(act_res,pred_res)