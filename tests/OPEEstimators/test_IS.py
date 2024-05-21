import unittest
from unittest.mock import MagicMock
import torch
import numpy as np
from offline_rl_ope.OPEEstimators.IS import ISEstimator
# from ..base import (test_reward_values, reward_test_res, weight_test_res,
#                     msk_test_res)
from ..base import (
    single_discrete_action_test as sdat,
    duel_discrete_action_test as ddat,
    bin_discrete_action_test as bdat
    )


gamma = 0.99

for test_conf in [sdat,ddat,bdat]:
    class ISEstimatorTest(unittest.TestCase):
        
        def setUp(self) -> None:
            self.is_estimator = ISEstimator(norm_weights=False)
        
        
        def test_get_traj_discnt_reward(self):
            for r in test_conf.test_reward_values:
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
                        test_conf.test_reward_values, test_conf.reward_test_res
                    )
                }
                return lkp["_".join([str(reward_array), str(discount)])]
            
            self.is_estimator.get_traj_discnt_reward = MagicMock(
                side_effect=__mock_return)
            
            rewards = [torch.Tensor(r) for r in test_conf.test_reward_values]
            pred_res = self.is_estimator.get_dataset_discnt_reward(
                rewards=rewards, discount=gamma, h=test_conf.reward_test_res.shape[1]
                )
            self.assertTrue(pred_res.shape, test_conf.reward_test_res.shape)
            np.testing.assert_allclose(pred_res.numpy(),test_conf.reward_test_res.numpy(),
                                    np.abs(test_conf.reward_test_res.mean().numpy()))
            
        
        def test_predict_traj_rewards(self):
            def __mock_return(rewards, discount, h):
                return test_conf.reward_test_res
            self.is_estimator.get_dataset_discnt_reward = MagicMock(
                side_effect=__mock_return)
            rewards = [torch.Tensor(r) for r in test_conf.test_reward_values]
            pred_res = self.is_estimator.predict_traj_rewards(
                rewards=rewards, actions=[], states=[], weights=test_conf.weight_test_res,
                discount=gamma, is_msk=test_conf.msk_test_res)
            test_res = np.multiply(
                test_conf.reward_test_res.numpy(), 
                test_conf.weight_test_res.numpy()/test_conf.weight_test_res.shape[0]
                )
            test_res=test_res.sum(axis=1)
            #test_res = test_res.sum(axis=1).mean()
            tol = test_res.mean()/1000
            self.assertEqual(pred_res.shape, torch.Size((len(rewards),)))
            np.testing.assert_allclose(pred_res.numpy(), test_res, atol=tol)
            
