import unittest
from unittest.mock import MagicMock
import torch
from typing import List, Literal
import numpy as np
from offline_rl_ope.api.StandardEstimators import VanillaDR, WDR
from offline_rl_ope.OPEEstimators.DirectMethod import DirectMethodBase
from parameterized import parameterized
from ..base import test_configs_fmt, get_wpd_denoms

gamma = 0.99

dm_model = MagicMock(spec=DirectMethodBase)

def dr(
    rewards:List[torch.Tensor], 
    states:List[torch.Tensor], 
    actions:List[torch.Tensor], 
    weights:List[torch.Tensor],
    is_msk:torch.Tensor, 
    sa_lst:List[List[float]], 
    s_lst:List[List[float]],
    is_type:Literal["is","wis","pd","wpd"]
    )->List[torch.Tensor]:
    traj_values = []
    weight_vals = []
    if is_type == "is":
        for w in weights:
            h = w.shape[0]
            weight_vals.append(w.prod(0,keepdim=True).repeat((h)))
    elif is_type == "wis":
        weight_vals = []
        for w in weights:
            h = w.shape[0]
            weight_vals.append(w.prod(0,keepdim=True).repeat((h)))
        max_h = max(len(w) for w in weights)
        test_res_w = get_wpd_denoms(
            weights=weights,h=max_h,is_type="is",agg_type="mean"
            )
        weight_vals = [w/d[:len(w)] for w,d in zip(weight_vals,test_res_w)]
    elif is_type == "pd":
        for w in weights:
            h = w.shape[0]
            weight_vals.append(w.cumprod(0))
    elif is_type == "wpd":
        weight_vals = []
        for w in weights:
            weight_vals.append(w.cumprod(0))
        max_h = max(len(w) for w in weights)
        test_res_w = get_wpd_denoms(weights=weights,h=max_h,agg_type="mean")
        weight_vals = [w/d[:len(w)] for w,d in zip(weight_vals,test_res_w)]
    else:
        raise ValueError
    for idx, (r,s,a,w,msk,qs,vs) in enumerate(zip(
        rewards, 
        states, 
        actions, 
        weight_vals, 
        is_msk,
        sa_lst, 
        s_lst
        )):
        qs = torch.tensor(qs).squeeze()
        vs = torch.tensor(vs).squeeze()
        r = r.squeeze()
        msk = msk.squeeze()
        w = w.squeeze()
        h = int(msk.sum().item())
        discnt = torch.tensor(gamma).repeat(h)
        discnt_pow = torch.arange(0,h)
        discnt = torch.pow(discnt,discnt_pow)
        _t1 = torch.mul(torch.mul(discnt,r),w)
        _t2 = torch.mul(w,qs)
        prev_weights = torch.roll(w,1)
        prev_weights[0] = torch.tensor([1])
        _t3 = torch.mul(prev_weights,vs)
        _t4 = torch.mul(discnt, _t2-_t3)
        traj_val = (_t1-_t4).sum()/len(rewards)
        traj_values.append(traj_val)
    return traj_values


class DREstimatorTest(unittest.TestCase):    
    
    @parameterized.expand(test_configs_fmt)
    def test_predict_dr(self, name, test_conf):
        traj_values = []
        rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
        weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
        states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
        actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
        max_h = max([len(w) for w in weights])
        is_msk = torch.concat([
            torch.concat(
                [torch.ones(x.shape[0]),torch.zeros(max_h-x.shape[0])]
                )[None,:] 
            for x in weights
            ], axis=0)
        traj_values = dr(
            rewards=rewards, states=states, actions=actions, weights=weights,
            is_msk=is_msk, s_lst=test_conf.test_dm_s_values,
            sa_lst=test_conf.test_dm_sa_values, is_type="is"
        )
        test_res = torch.concat(
            [v[None] for v in traj_values]).sum()
            
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
        
        
        weight_tens = torch.concat([
            torch.concat([x, torch.ones(max_h-x.shape[0])])[None,:] for x in 
            weights
        ], axis=0)
        weight_tens = weight_tens.prod(1,keepdim=True).repeat(
            (1,is_msk.shape[1]))
        
        is_est = VanillaDR(
            dm_model=dm_model, 
            clip=0.0
            )
        pred_res = is_est.predict(
            rewards=rewards, states=states, actions=actions, 
            weights=weight_tens, discount=gamma, is_msk=is_msk
        )
        tol = (np.abs(test_res/100)).item()
        np.testing.assert_allclose(pred_res.numpy(),test_res.numpy(), atol=tol)
        
    @parameterized.expand(test_configs_fmt)
    def test_predict_wis(self, name, test_conf):
        traj_values = []
        rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
        weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
        states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
        actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
        max_h = max([len(w) for w in weights])
        is_msk = torch.concat([
            torch.concat(
                [torch.ones(x.shape[0]),torch.zeros(max_h-x.shape[0])]
                )[None,:] 
            for x in weights
            ], axis=0)
        traj_values = dr(
            rewards=rewards, states=states, actions=actions, weights=weights,
            is_msk=is_msk, s_lst=test_conf.test_dm_s_values,
            sa_lst=test_conf.test_dm_sa_values, is_type="wis"
        )
        test_res = torch.concat(
            [v[None] for v in traj_values]).sum()
            
        def q_side_effect(state:torch.Tensor, action:torch.Tensor):
            lkp = {
                "_".join([str(torch.Tensor(s)), str(torch.Tensor(a))]): q
                for s,a,q in zip(test_conf.test_state_vals, 
                                 test_conf.test_action_vals, 
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
        
        
        weight_tens = torch.concat([
            torch.concat([x, torch.ones(max_h-x.shape[0])])[None,:] for x in 
            weights
        ], axis=0)
        weight_tens = weight_tens.prod(1,keepdim=True).repeat(
            (1,is_msk.shape[1]))
        
        is_est = WDR(
            dm_model=dm_model, 
            clip=0.0
            )
        pred_res = is_est.predict(
            rewards=rewards, states=states, actions=actions, 
            weights=weight_tens, discount=gamma, is_msk=is_msk
        )
        tol = (np.abs(test_res/100)).item()
        np.testing.assert_allclose(pred_res.numpy(),test_res.numpy(), atol=tol)
        
        
    @parameterized.expand(test_configs_fmt)
    def test_predict_pd(self, name, test_conf):
        traj_values = []
        rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
        weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
        states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
        actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
        max_h = max([len(w) for w in weights])
        is_msk = torch.concat([
            torch.concat(
                [torch.ones(x.shape[0]),torch.zeros(max_h-x.shape[0])]
                )[None,:] 
            for x in weights
            ], axis=0)
        traj_values = dr(
            rewards=rewards, states=states, actions=actions, weights=weights,
            is_msk=is_msk, s_lst=test_conf.test_dm_s_values,
            sa_lst=test_conf.test_dm_sa_values, is_type="pd"
        )
        test_res = torch.concat(
            [v[None] for v in traj_values]).sum()
            
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
        
        
        weight_tens = torch.concat([
            torch.concat([x, torch.ones(max_h-x.shape[0])])[None,:] for x in 
            weights
        ], axis=0)
        weight_tens = weight_tens.cumprod(1)
        
        is_est = VanillaDR(
            dm_model=dm_model, 
            clip=0.0
            )
        pred_res = is_est.predict(
            rewards=rewards, states=states, actions=actions, 
            weights=weight_tens, discount=gamma, is_msk=is_msk
        )
        tol = (np.abs(test_res/100)).item()
        np.testing.assert_allclose(pred_res.numpy(),test_res.numpy(), atol=tol)

    @parameterized.expand(test_configs_fmt)
    def test_predict_wpd(self, name, test_conf):
        traj_values = []
        rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
        weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
        states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
        actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
        max_h = max([len(w) for w in weights])
        is_msk = torch.concat([
            torch.concat(
                [torch.ones(x.shape[0]),torch.zeros(max_h-x.shape[0])]
                )[None,:] 
            for x in weights
            ], axis=0)
        traj_values = dr(
            rewards=rewards, states=states, actions=actions, weights=weights,
            is_msk=is_msk, s_lst=test_conf.test_dm_s_values,
            sa_lst=test_conf.test_dm_sa_values, is_type="wpd"
        )
        test_res = torch.concat(
            [v[None] for v in traj_values]).sum()
            
        def q_side_effect(state:torch.Tensor, action:torch.Tensor):
            lkp = {
                "_".join([str(torch.Tensor(s)), str(torch.Tensor(a))]): q
                for s,a,q in zip(test_conf.test_state_vals, 
                                 test_conf.test_action_vals, 
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
        
        
        weight_tens = torch.concat([
            torch.concat([x, torch.ones(max_h-x.shape[0])])[None,:] for x in 
            weights
        ], axis=0)
        weight_tens = weight_tens.cumprod(1)
        
        is_est = WDR(
            dm_model=dm_model, 
            clip=0.0,
            )
        pred_res = is_est.predict(
            rewards=rewards, states=states, actions=actions, 
            weights=weight_tens, discount=gamma, is_msk=is_msk
        )
        tol = (np.abs(test_res/100)).item()
        np.testing.assert_allclose(pred_res.numpy(),test_res.numpy(), atol=tol)

