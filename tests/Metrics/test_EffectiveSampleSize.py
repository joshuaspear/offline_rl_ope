import unittest
import torch
import numpy as np
from offline_rl_ope.Metrics import EffectiveSampleSize
from offline_rl_ope import logger
from parameterized import parameterized_class
from ..base import test_configs_fmt_class, TestConfig

@parameterized_class(test_configs_fmt_class)
class EffectiveSampleSizeTest(unittest.TestCase):

    test_conf:TestConfig
    
    def test_call(self):
        num = 2
        weights = self.test_conf.weight_test_res.sum(dim=1)
        assert len(weights) == 2
        denum = 1 + torch.var(weights)
        act_res = (num/denum).item()
        metric = EffectiveSampleSize(nan_if_all_0=True)
        pred_res = metric(
            weights=self.test_conf.weight_test_res
        )
        tol = act_res/1000
        np.testing.assert_allclose(pred_res, act_res, atol=tol)