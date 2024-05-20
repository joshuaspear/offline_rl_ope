import unittest
import torch
import numpy as np
from offline_rl_ope.Metrics import EffectiveSampleSize
from offline_rl_ope import logger
# from ..base import weight_test_res
from ..base import (
    single_discrete_action_test as sdat,
    duel_discrete_action_test as ddat,
    bin_discrete_action_test as bdat
    )

for test_conf in [sdat,ddat,bdat]:
    class TestImportanceSampler:
        
        def __init__(self) -> None:
            self.is_weight_calc = None
            self.traj_is_weights = test_conf.weight_test_res
            

    class EffectiveSampleSizeTest(unittest.TestCase):

        def test_call(self):
            num = 2
            weights = test_conf.weight_test_res.sum(dim=1)
            assert len(weights) == 2
            denum = 1 + torch.var(weights)
            act_res = (num/denum).item()
            metric = EffectiveSampleSize(nan_if_all_0=True)
            pred_res = metric(
                weights=test_conf.weight_test_res
            )
            tol = act_res/1000
            np.testing.assert_allclose(pred_res, act_res, atol=tol)