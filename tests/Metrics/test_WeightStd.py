import unittest
import torch
import numpy as np
import copy
from offline_rl_ope.Metrics import WeightStd
from offline_rl_ope import logger
from parameterized import parameterized_class
from ..base import test_configs_fmt_class, TestConfig

@parameterized_class(test_configs_fmt_class)
class TestWeightStd(unittest.TestCase):

    test_conf:TestConfig
    
    def test_call(self):
        fnl_weights = []
        for idx,i in enumerate(self.test_conf.traj_lengths):
            fnl_weights.append(
                self.test_conf.weight_test_res[idx,:i].sum(
                    dim=0,
                    keepdim=True
                    )
                )
        fnl_weights_tens = torch.concat(fnl_weights, axis=0)
        act_res = torch.std(fnl_weights_tens).item()
        metric = WeightStd()
        pred_res = metric(
            weights=self.test_conf.weight_test_res, 
            weight_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(act_res,pred_res)