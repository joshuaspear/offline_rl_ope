import unittest
from unittest.mock import MagicMock
import torch
import numpy as np
from offline_rl_ope.OPEEstimators.DoublyRobust import DREstimator 
from offline_rl_ope.OPEEstimators.DirectMethod import DirectMethodBase
from offline_rl_ope.RuntimeChecks import check_array_dim
from parameterized import parameterized
from ..base import test_configs_fmt


gamma = 0.99

dm_model = MagicMock(spec=DirectMethodBase)

class DREstimatorTest(unittest.TestCase):
    
    @parameterized.expand(test_configs_fmt)
    def test_update_step_ignore(self, name, test_conf):
        
        # is_est = DREstimator(dm_model=MockDMModel(), norm_weights=False, 
        #                     clip=None, ignore_nan=True)
        is_est = DREstimator(dm_model=dm_model, norm_weights=False, 
                                clip=0.0, ignore_nan=True)
        v_dr_t = torch.tensor([0.0])
        v_t = torch.tensor(test_conf.test_dm_s_values[0][-1])
        p_t = test_conf.weight_test_res[0,-1].reshape(-1)
        r_t = torch.tensor(test_conf.test_reward_values[0][-1]).float()
        q_t = torch.tensor(test_conf.test_dm_sa_values[0][-1])
        assert len(v_dr_t.shape) == 1, "Test input dim not correct"
        assert len(v_t.shape) == 1, "Test input dim not correct"
        assert len(p_t.shape) == 1, "Test input dim not correct"
        assert len(r_t.shape) == 1, "Test input dim not correct"
        assert len(q_t.shape) == 1, "Test input dim not correct"
        pred_res:torch.Tensor = is_est._DREstimator__update_step(
            v_t, p_t, r_t, v_dr_t, torch.tensor([gamma]), q_t
            )
        test_res:torch.Tensor = v_t + p_t*(r_t+torch.tensor(gamma)*v_dr_t-q_t)
        tol = test_res/1000
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=tol.numpy().item())
    
    @parameterized.expand(test_configs_fmt)
    def test_get_traj_discnt_reward(self, name, test_conf):
        # dm_model = MockDMModel()
        def q_side_effect(state:torch.Tensor, action:torch.Tensor):
            lkp = {
                "_".join([str(torch.Tensor(s)), str(torch.Tensor(a))]): q
                for s,a,q in zip(test_conf.test_state_vals, test_conf.test_action_vals, 
                                test_conf.test_dm_sa_values)
            }
            res = lkp["_".join([str(state), str(action)])]
            return torch.Tensor(res)
        def v_side_effect(state:torch.Tensor):
            lkp = {
                str(torch.Tensor(s)): v 
                for s,v in zip(test_conf.test_state_vals, test_conf.test_dm_s_values)
            }
            res = lkp[str(state)]
            return torch.Tensor(res)
        dm_model.get_q = MagicMock(side_effect=q_side_effect)
        dm_model.get_v = MagicMock(side_effect=v_side_effect)
        # dm_model.get_q.return_value = q_side_effect
        # dm_model.get_v.return_value = v_side_effect
        is_est = DREstimator(dm_model=dm_model, norm_weights=False, clip=0.0, 
                            ignore_nan=True)
        pred_res = []
        test_res = []
        for idx, traj in enumerate(zip(
            test_conf.test_state_vals, test_conf.weight_test_res, test_conf.test_reward_values, 
            test_conf.test_action_vals, test_conf.test_dm_sa_values, 
            test_conf.test_dm_s_values, test_conf.msk_test_res
            )):
            s_t = torch.Tensor(traj[0])
            p_t = torch.masked_select(traj[1], traj[6]>0).reshape(-1,1)
            r_t = torch.Tensor(traj[2]).float()
            a_t = torch.Tensor(traj[3])
            q_t = torch.Tensor(traj[4])
            v_t = torch.Tensor(traj[5])
            assert len(s_t.shape) == 2, "Test input dim not correct"
            assert len(p_t.shape) == 2, "Test input dim not correct"
            assert len(r_t.shape) == 2, "Test input dim not correct"
            assert len(a_t.shape) == 2, "Test input dim not correct"
            assert len(q_t.shape) == 2, "Test input dim not correct"
            assert len(v_t.shape) == 2, "Test input dim not correct"
            __pred_res = is_est.get_traj_discnt_reward(
                reward_array=r_t, discount=gamma, 
                state_array=s_t, action_array=a_t, weight_array=p_t)
            pred_res.append(__pred_res.numpy())
            __test_res_v = torch.tensor([0.0])
            assert len(__test_res_v.shape) == 1, "Test input dim not correct"
            for i in np.arange(s_t.shape[0]-1, 0-1, -1):
                _v_t_i = v_t[i]
                _q_t_i = q_t[i]
                _p_t_i = p_t[i]
                _r_t_i = r_t[i]
                _gamma = torch.tensor([gamma])
                assert len(_v_t_i.shape) == 1, "Test input dim not correct"
                assert len(_p_t_i.shape) == 1, "Test input dim not correct"
                assert len(_r_t_i.shape) == 1, "Test input dim not correct"
                assert len(_q_t_i.shape) == 1, "Test input dim not correct"
                assert len(_gamma.shape) == 1, "Test input dim not correct"
                __test_res_v = is_est._DREstimator__update_step(
                    v_t=v_t[i], q_t=q_t[i], p_t=p_t[i], r_t=r_t[i], 
                    gamma=_gamma, v_dr_t=__test_res_v)
            test_res.append(__test_res_v.numpy())
        pred_res = np.concatenate(pred_res)
        test_res = np.concatenate(test_res)
        tol = (test_res.mean()/1000).item()
        np.testing.assert_allclose(pred_res, test_res, atol=tol)
    
    @parameterized.expand(test_configs_fmt)
    def test_predict_traj_rewards(self, name, test_conf):
        #dm_model = MockDMModel()
        def q_side_effect(state:torch.Tensor, action:torch.Tensor):
            lkp = {
                "_".join([str(torch.Tensor(s)), str(torch.Tensor(a))]): q
                for s,a,q in zip(test_conf.test_state_vals, test_conf.test_action_vals, 
                                test_conf.test_dm_sa_values)
            }
            res = lkp["_".join([str(state), str(action)])]
            return torch.Tensor(res)
        def v_side_effect(state:torch.Tensor):
            lkp = {
                str(torch.Tensor(s)): v 
                for s,v in zip(test_conf.test_state_vals, test_conf.test_dm_s_values)
            }
            res = lkp[str(state)]
            return torch.Tensor(res)
        dm_model.get_q = MagicMock(side_effect=q_side_effect)
        dm_model.get_v = MagicMock(side_effect=v_side_effect)
        # dm_model.get_q.return_value = q_side_effect
        # dm_model.get_v.return_value = v_side_effect
        is_est = DREstimator(dm_model=dm_model, norm_weights=False, clip=0.0, 
                            ignore_nan=True)
        rewards = [
            torch.Tensor(x).float() for x in test_conf.test_reward_values
            ]
        states = [torch.Tensor(x) for x in test_conf.test_state_vals]
        actions = [torch.Tensor(x) for x in test_conf.test_action_vals]
        test_res = []
        pred_res = is_est.predict_traj_rewards(
            rewards=rewards, states=states, actions=actions, 
            weights=test_conf.weight_test_res, discount=gamma, 
            is_msk=test_conf.msk_test_res
            )
        #weight_test_res = weight_test_res/weight_test_res.shape[0]
        denom = test_conf.weight_test_res.shape[0]
        for idx, (r,s,a,w,msk) in enumerate(zip(
            rewards, states, actions, test_conf.weight_test_res, test_conf.msk_test_res
            )):
            w = w/denom
            p = torch.masked_select(w, msk>0).reshape(-1,1)
            assert len(r.shape) == 2, "Test input dim not correct"
            assert len(s.shape) == 2, "Test input dim not correct"
            assert len(a.shape) == 2, "Test input dim not correct"
            assert len(p.shape) == 2, "Test input dim not correct"
            assert isinstance(gamma, float), "Test input dim not correct"
            __test_res = is_est.get_traj_discnt_reward(
                reward_array=r, discount=gamma, state_array=s, 
                action_array=a, 
                weight_array=p)
            test_res.append(__test_res.numpy())
        #test_res = np.concatenate(test_res).mean()
        test_res = np.concatenate(test_res)
        tol = (np.abs(test_res.mean()/100)).item()
        self.assertEqual(pred_res.shape, torch.Size((len(rewards),)))
        np.testing.assert_allclose(pred_res.numpy(),test_res, atol=tol)    
            
    