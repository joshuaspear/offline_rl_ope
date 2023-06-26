import copy
import numpy as np
import torch
import unittest
from offline_rl_ope.OPEEstimators.utils import (
    clip_weights, clip_weights_pass, norm_weights_pass, wis_norm_weights)
from ..base import (weight_test_res, msk_test_res)



class UtilsTest(unittest.TestCase):
    
    def setUp(self) -> None:
        self.clip_toll = weight_test_res.numpy().mean()/1000
    
    def test_clip_weights(self):
        clip = 1.2
        test_res = weight_test_res.clamp(max=1.2, min=1/1.2)
        pred_res = clip_weights(weight_test_res, clip=clip)
        self.assertEqual(pred_res.shape,weight_test_res.shape)
        np.testing.assert_allclose(pred_res, test_res, atol=self.clip_toll)
        
    def test_clip_weights_pass(self):
        clip = 1.2
        test_res = copy.deepcopy(weight_test_res)
        pred_res = clip_weights_pass(weight_test_res, clip=clip)
        self.assertEqual(pred_res.shape,weight_test_res.shape)
        np.testing.assert_allclose(pred_res, test_res, atol=self.clip_toll)
        
    # def test_norm_weights_pass(self):
    #     test_res = weight_test_res/msk_test_res.sum(axis=0)
    #     toll = test_res.mean()/1000
    #     pred_res = norm_weights_pass(traj_is_weights=weight_test_res, 
    #                                  is_msk=msk_test_res)
    #     self.assertEqual(pred_res.shape,weight_test_res.shape)
    #     np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
    #                                atol=toll.numpy())
    
    def test_norm_weights_pass(self):
            test_res = copy.deepcopy(weight_test_res)
            toll = test_res.mean()/1000
            pred_res = norm_weights_pass(traj_is_weights=weight_test_res, 
                                         is_msk=msk_test_res)
            self.assertEqual(pred_res.shape,weight_test_res.shape)
            np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                       atol=toll.numpy())
    
    def test_norm_weights_wis(self):
        test_res = weight_test_res/weight_test_res.sum(axis=0)
        toll = test_res.mean()/1000
        pred_res = wis_norm_weights(traj_is_weights=weight_test_res, 
                                    is_msk=msk_test_res)
        self.assertEqual(pred_res.shape,weight_test_res.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                   atol=toll.numpy())

        
        
