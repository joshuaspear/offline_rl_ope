import unittest
from unittest.mock import MagicMock
import torch
import logging
import numpy as np
from offline_rl_ope.OPEEstimators.IS import ISEstimator, ISEstimatorBase
from ..base import (test_reward_values, reward_test_res, weight_test_res,
                    msk_test_res)

gamma = 0.99

class ISEstimatorTest(unittest.TestCase):
    
    def setUp(self) -> None:
        self.is_estimator = ISEstimator(norm_weights=False, clip=None)
    
    
    def test_get_traj_discnt_reward(self):
        for r in test_reward_values:
            disc_vals = torch.full(size=(len(r),1), fill_value=gamma)
            power_vals = torch.Tensor(list(range(0,len(r)))).view(-1,1)
            disc_vals = torch.pow(disc_vals,power_vals).squeeze()
            r = torch.Tensor(r).view(-1,1)
            test_res = r.squeeze()*disc_vals
            tol = np.abs(test_res.mean().numpy().item())
            res = self.is_estimator.get_traj_discnt_reward(
                reward_array=r, discount=gamma)
            self.assertEqual(res.shape,torch.Size((len(r),)))
            np.testing.assert_allclose(res, test_res, atol=tol)
    
    def test_get_dataset_discnt_reward(self):
        def __mock_return(reward_array, discount):
            lkp = {
                "_".join([str(torch.Tensor(r)), str(gamma)]): w for r,w in zip(
                    test_reward_values, reward_test_res
                )
            }
            return lkp["_".join([str(reward_array), str(discount)])]
        
        self.is_estimator.get_traj_discnt_reward = MagicMock(
            side_effect=__mock_return)
        
        rewards = [torch.Tensor(r) for r in test_reward_values]
        pred_res = self.is_estimator.get_dataset_discnt_reward(
            rewards=rewards, discount=gamma, h=reward_test_res.shape[1]
            )
        self.assertTrue(pred_res.shape, reward_test_res.shape)
        np.testing.assert_allclose(pred_res.numpy(),reward_test_res.numpy(),
                                   np.abs(reward_test_res.mean().numpy()))
        
    
    def test_predict_traj_rewards(self):
        def __mock_return(rewards, discount, h):
            return reward_test_res
        self.is_estimator.get_dataset_discnt_reward = MagicMock(
            side_effect=__mock_return)
        rewards = [torch.Tensor(r) for r in test_reward_values]
        pred_res = self.is_estimator.predict_traj_rewards(
            rewards=rewards, actions=[], states=[], weights=weight_test_res,
            discount=gamma, is_msk=msk_test_res)
        test_res = np.multiply(
            reward_test_res.numpy(), 
            weight_test_res.numpy()/weight_test_res.shape[0]
            )
        test_res=test_res.sum(axis=1)
        #test_res = test_res.sum(axis=1).mean()
        tol = test_res.mean()/1000
        self.assertEqual(pred_res.shape, torch.Size((len(rewards),)))
        np.testing.assert_allclose(pred_res.numpy(), test_res, atol=tol)
        
