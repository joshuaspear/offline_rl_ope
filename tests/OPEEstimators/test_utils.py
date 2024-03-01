import copy
import numpy as np
import torch
import unittest
from offline_rl_ope.OPEEstimators.utils import (
    clip_weights, clip_weights_pass, VanillaNormWeights, WISWeightNorm)
from ..base import (weight_test_res, msk_test_res)

weight_test_res_alter = copy.deepcopy(weight_test_res)
weight_test_res_alter[0] = torch.zeros(len(weight_test_res_alter[0]))

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
    
    def test_norm_weights_vanilla(self):
            denom = weight_test_res.shape[0]
            test_res = weight_test_res/denom
            toll = test_res.mean()/1000
            calculator = VanillaNormWeights()
            pred_res = calculator(traj_is_weights=weight_test_res, 
                                  is_msk=msk_test_res)
            self.assertEqual(pred_res.shape,weight_test_res.shape)
            np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                       atol=toll.numpy())
    
    def test_norm_weights_wis(self):
        denom = weight_test_res.sum(dim=0)
        test_res = weight_test_res/denom
        toll = test_res.mean()/1000
        calculator = WISWeightNorm()
        pred_res = calculator(traj_is_weights=weight_test_res, 
                              is_msk=msk_test_res)
        self.assertEqual(pred_res.shape,weight_test_res.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                   atol=toll.numpy())

    def test_norm_weights_wis_smooth(self):
        smooth_eps = 0.00000001
        denom = weight_test_res_alter.sum(dim=0)+smooth_eps
        test_res = weight_test_res_alter/denom
        toll = test_res.nanmean()/1000
        calculator = WISWeightNorm(smooth_eps=smooth_eps)
        pred_res = calculator(traj_is_weights=weight_test_res_alter, 
                              is_msk=msk_test_res)
        self.assertEqual(pred_res.shape,weight_test_res_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                   atol=toll.numpy())
        
    def test_norm_weights_wis_no_smooth(self):
        denom = weight_test_res_alter.sum(dim=0)
        test_res = weight_test_res_alter/denom
        toll = test_res.nanmean()/1000
        calculator = WISWeightNorm()
        pred_res = calculator(traj_is_weights=weight_test_res_alter, 
                              is_msk=msk_test_res)
        self.assertEqual(pred_res.shape,weight_test_res_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                   atol=toll.numpy(), equal_nan=True)
        
    def test_norm_weights_wis_smooth_discount(self):
        smooth_eps = 0.00000001
        discount=0.99
        discnt_tens = torch.full(
            weight_test_res_alter.shape,
            discount
            )
        discnt_pows = torch.arange(
            0, weight_test_res_alter.shape[1])[None,:].repeat(
                weight_test_res_alter.shape[0],1
                )
        discnt_tens = torch.pow(discnt_tens,discnt_pows)
        denom = torch.mul(
            weight_test_res_alter,
            discnt_tens
        )
        denom = denom.sum(dim=0)+smooth_eps
        test_res = weight_test_res_alter/denom
        toll = test_res.nanmean()/1000
        calculator = WISWeightNorm(
            smooth_eps=smooth_eps,
            discount=discount
            )
        pred_res = calculator(traj_is_weights=weight_test_res_alter, 
                              is_msk=msk_test_res)
        self.assertEqual(pred_res.shape,weight_test_res_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                   atol=toll.numpy())

    def test_norm_weights_wis_no_smooth_discount(self):
        discount=0.99
        discnt_tens = torch.full(
            weight_test_res_alter.shape,
            discount
            )
        discnt_pows = torch.arange(
            0, weight_test_res_alter.shape[1])[None,:].repeat(
                weight_test_res_alter.shape[0],1
                )
        discnt_tens = torch.pow(discnt_tens,discnt_pows)
        denom = torch.mul(
            weight_test_res_alter,
            discnt_tens
        )
        denom = denom.sum(dim=0)
        test_res = weight_test_res_alter/denom
        toll = test_res.nanmean()/1000
        calculator = WISWeightNorm(
            discount=discount
            )
        pred_res = calculator(traj_is_weights=weight_test_res_alter, 
                              is_msk=msk_test_res)
        self.assertEqual(pred_res.shape,weight_test_res_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                   atol=toll.numpy())

    def test_norm_weights_wis_smooth_avg(self):
        smooth_eps = 0.00000001
        time_t_freq = msk_test_res.sum(dim=0, keepdim=True).repeat(
            msk_test_res.shape[0],1
        )
        denom = weight_test_res_alter/time_t_freq
        denom = denom.sum(dim=0)+smooth_eps
        test_res = weight_test_res_alter/denom
        toll = test_res.nanmean()/1000
        calculator = WISWeightNorm(
            smooth_eps=smooth_eps,
            avg_denom=True
            )
        pred_res = calculator(traj_is_weights=weight_test_res_alter, 
                              is_msk=msk_test_res)
        self.assertEqual(pred_res.shape,weight_test_res_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                   atol=toll.numpy())

    def test_norm_weights_wis_no_smooth_avg(self):
        time_t_freq = msk_test_res.sum(dim=0, keepdim=True).repeat(
            msk_test_res.shape[0],1
        )
        denom = weight_test_res_alter/time_t_freq
        denom = denom.sum(dim=0)
        test_res = weight_test_res_alter/denom
        toll = test_res.nanmean()/1000
        calculator = WISWeightNorm(
            avg_denom=True
            )
        pred_res = calculator(traj_is_weights=weight_test_res_alter, 
                              is_msk=msk_test_res)
        self.assertEqual(pred_res.shape,weight_test_res_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                   atol=toll.numpy())

    def test_norm_weights_wis_smooth_discount_avg(self):
        smooth_eps = 0.00000001
        discount=0.99
        discnt_tens = torch.full(
            weight_test_res_alter.shape,
            discount
            )
        discnt_pows = torch.arange(
            0, weight_test_res_alter.shape[1])[None,:].repeat(
                weight_test_res_alter.shape[0],1
                )
        discnt_tens = torch.pow(discnt_tens,discnt_pows)
        denom = torch.mul(
            weight_test_res_alter,
            discnt_tens
        )
        time_t_freq = msk_test_res.sum(dim=0, keepdim=True).repeat(
            msk_test_res.shape[0],1
        )
        denom = denom/time_t_freq
        denom = denom.sum(dim=0)+smooth_eps
        test_res = weight_test_res_alter/denom
        toll = test_res.nanmean()/1000
        calculator = WISWeightNorm(
            smooth_eps=smooth_eps,
            discount=discount,
            avg_denom=True
            )
        pred_res = calculator(traj_is_weights=weight_test_res_alter, 
                              is_msk=msk_test_res)
        self.assertEqual(pred_res.shape,weight_test_res_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                   atol=toll.numpy())

    def test_norm_weights_wis_no_smooth_discount_avg(self):
        discount=0.99
        discnt_tens = torch.full(
            weight_test_res_alter.shape,
            discount
            )
        discnt_pows = torch.arange(
            0, weight_test_res_alter.shape[1])[None,:].repeat(
                weight_test_res_alter.shape[0],1
                )
        discnt_tens = torch.pow(discnt_tens,discnt_pows)
        denom = torch.mul(
            weight_test_res_alter,
            discnt_tens
        )
        time_t_freq = msk_test_res.sum(dim=0, keepdim=True).repeat(
            msk_test_res.shape[0],1
        )
        denom = denom/time_t_freq
        denom = denom.sum(dim=0)
        test_res = weight_test_res_alter/denom
        toll = test_res.nanmean()/1000
        calculator = WISWeightNorm(
            discount=0.99,
            avg_denom=True
            )
        pred_res = calculator(traj_is_weights=weight_test_res_alter, 
                              is_msk=msk_test_res)
        self.assertEqual(pred_res.shape,weight_test_res_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                   atol=toll.numpy())
