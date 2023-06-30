import unittest
from unittest.mock import MagicMock
import torch
import logging
import numpy as np
from offline_rl_ope.OPEEstimators.DoublyRobust import DREstimator 
from ..base import (test_reward_values, weight_test_res, test_dm_s_values, 
                    test_dm_sa_values, test_state_vals, test_action_vals, 
                    msk_test_res)

gamma = 0.99


class MockDMModel:
    
    def __init__(self) -> None:
        pass 
    
    def get_v(self, *args, **kwargs):
        pass
    
    def get_q(self, *args, **kwargs):
        pass

class DREstimatorTest(unittest.TestCase):
    
    def test_update_step_ignore(self):
        
        is_est = DREstimator(dm_model=MockDMModel(), norm_weights=False, 
                             clip=None, ignore_nan=True)
        v_dr_t = torch.tensor(0)
        v_t = torch.tensor(test_dm_s_values[0][-1])
        p_t = weight_test_res[0,-1]
        r_t = torch.tensor(test_reward_values[0][-1])
        q_t = torch.tensor(test_dm_sa_values[0][-1])
        pred_res:torch.Tensor = is_est._DREstimator__update_step(
            v_t, p_t, r_t, v_dr_t, gamma, q_t
            )
        test_res:torch.Tensor = v_t + p_t*(r_t+torch.tensor(gamma)*v_dr_t-q_t)
        tol = test_res/1000
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                   atol=tol.numpy().item())
    
    def test_get_traj_discnt_reward(self):
        dm_model = MockDMModel()
        def q_side_effect(state:torch.Tensor, action:torch.Tensor):
            lkp = {
                "_".join([str(torch.Tensor(s)), str(torch.Tensor(a))]): q
                for s,a,q in zip(test_state_vals, test_action_vals, 
                                 test_dm_sa_values)
            }
            res = lkp["_".join([str(state), str(action)])]
            return torch.Tensor(res)
        def v_side_effect(state:torch.Tensor):
            lkp = {
                str(torch.Tensor(s)): v 
                for s,v in zip(test_state_vals, test_dm_s_values)
            }
            res = lkp[str(state)]
            return torch.Tensor(res)
        dm_model.get_q = MagicMock(side_effect=q_side_effect)
        dm_model.get_v = MagicMock(side_effect=v_side_effect)
        is_est = DREstimator(dm_model=dm_model, norm_weights=False, clip=None, 
                             ignore_nan=True)
        pred_res = []
        test_res = []
        for idx, traj in enumerate(zip(test_state_vals, weight_test_res, 
                                       test_reward_values, test_action_vals, 
                                       test_dm_sa_values, test_dm_s_values, 
                                       msk_test_res)):
            s_t = torch.Tensor(traj[0])
            p_t = torch.masked_select(traj[1], traj[6]>0)
            r_t = torch.Tensor(traj[2])
            a_t = torch.Tensor(traj[3])
            q_t = torch.Tensor(traj[4])
            v_t = torch.Tensor(traj[5])
            __pred_res = is_est.get_traj_discnt_reward(
                reward_array=r_t, discount=gamma, state_array=s_t, 
                action_array=a_t, weight_array=p_t)
            pred_res.append(__pred_res.numpy())
            __test_res_v = torch.tensor(0)
            for i in np.arange(s_t.shape[0]-1, 0-1, -1):
                __test_res_v = is_est._DREstimator__update_step(
                    v_t=v_t[i], q_t=q_t[i], p_t=p_t[i], r_t=r_t[i], 
                    gamma=torch.tensor(gamma), v_dr_t=__test_res_v)
            test_res.append(__test_res_v.numpy())
        pred_res = np.concatenate(pred_res)
        test_res = np.concatenate(test_res)
        tol = (test_res.mean()/1000).item()
        np.testing.assert_allclose(pred_res, test_res, atol=tol)
        
    def test_predict(self):
        dm_model = MockDMModel()
        def q_side_effect(state:torch.Tensor, action:torch.Tensor):
            lkp = {
                "_".join([str(torch.Tensor(s)), str(torch.Tensor(a))]): q
                for s,a,q in zip(test_state_vals, test_action_vals, 
                                 test_dm_sa_values)
            }
            res = lkp["_".join([str(state), str(action)])]
            return torch.Tensor(res)
        def v_side_effect(state:torch.Tensor):
            lkp = {
                str(torch.Tensor(s)): v 
                for s,v in zip(test_state_vals, test_dm_s_values)
            }
            res = lkp[str(state)]
            return torch.Tensor(res)
        dm_model.get_q = MagicMock(side_effect=q_side_effect)
        dm_model.get_v = MagicMock(side_effect=v_side_effect)
        is_est = DREstimator(dm_model=dm_model, norm_weights=False, clip=None, 
                             ignore_nan=True)
        rewards = [torch.Tensor(x) for x in test_reward_values]
        states = [torch.Tensor(x) for x in test_state_vals]
        actions = [torch.Tensor(x) for x in test_action_vals]
        
        test_res = []
        pred_res = is_est.predict(rewards=rewards, states=states, 
                                  actions=actions, weights=weight_test_res, 
                                  discount=gamma, is_msk=msk_test_res)
        for idx, (r,s,a,w,msk) in enumerate(zip(rewards, states, actions, 
                                                weight_test_res, msk_test_res)):
            p = torch.masked_select(w, msk>0)
            __test_res = is_est.get_traj_discnt_reward(
                reward_array=r, discount=gamma, state_array=s, action_array=a, 
                weight_array=p)
            test_res.append(__test_res.numpy())
        test_res = np.concatenate(test_res).mean()
        tol = (test_res/1000).item()
        np.testing.assert_allclose(pred_res.numpy(),test_res, atol=tol)
        
            
    