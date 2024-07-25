import copy
import numpy as np
import torch
import unittest
from offline_rl_ope.OPEEstimators.utils import (
    clip_weights, clip_weights_pass, 
    get_traj_weight_final
    )
from parameterized import parameterized_class
from ..base import test_configs_fmt_class, TestConfig

@parameterized_class(test_configs_fmt_class)
class UtilsTestVanillaIS(unittest.TestCase):
    
    test_conf:TestConfig
    
    def setUp(self) -> None:
        self.clip_toll = self.test_conf.weight_test_res.numpy().mean()/1000

    def test_clip_weights(self):
        clip = 1.2
        test_res = self.test_conf.weight_test_res.clamp(max=1.2, min=1/1.2)
        assert len(self.test_conf.weight_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = clip_weights(self.test_conf.weight_test_res, clip=clip)
        self.assertEqual(pred_res.shape,self.test_conf.weight_test_res.shape)
        np.testing.assert_allclose(pred_res, test_res, atol=self.clip_toll)
    
    def test_clip_weights_pass(self):
        clip = 1.2
        test_res = copy.deepcopy(self.test_conf.weight_test_res)
        assert len(self.test_conf.weight_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = clip_weights_pass(self.test_conf.weight_test_res, clip=clip)
        self.assertEqual(pred_res.shape,self.test_conf.weight_test_res.shape)
        np.testing.assert_allclose(pred_res, test_res, atol=self.clip_toll)
    
    def test_get_traj_weight_final(self):
        pred_res = get_traj_weight_final(
            weights=self.test_conf.weight_test_res,
            is_msk=self.test_conf.msk_test_res
        )
        test_res = []
        for w in self.test_conf.test_act_indiv_weights:
            test_res.append(torch.tensor(w[-1])[None])
        test_res = torch.concat(test_res)
        np.testing.assert_allclose(pred_res, test_res, atol=0)
