import unittest
from unittest.mock import MagicMock
import torch
import numpy as np
import copy
from offline_rl_ope.components.Policy import Policy
from offline_rl_ope.components.ImportanceSampler import (
    VanillaIS, PerDecisionIS, ISWeightCalculator
    )
from offline_rl_ope import logger
from parameterized import parameterized_class
from ..base import test_configs_fmt_class, TestConfig


test_act_inidiv_rew = [
    np.array([1, -1*0.99, 1*(np.power(0.99,2)), 1*(np.power(0.99,3))]),
    np.array([-1, -1*0.99, -1*(np.power(0.99,2))])
    ]

class TestPolicy:
    
    def __init__(self, values) -> None:
        self.idx=0
        self.values = values
        
    def __call__(self, state: torch.Tensor, action: torch.Tensor):
        res = self.values[self.idx]
        self.idx += 1
        return torch.Tensor(res)
    
    def reset(self):
        self.idx = 0

@parameterized_class(test_configs_fmt_class)
class ISWeightCalculatorTest(unittest.TestCase):
    
    test_conf:TestConfig

    def setUp(self) -> None:

        self.test_act_norm_conts = [val.prod() for val in self.test_conf.test_act_indiv_weights]

        self.test_act_pd_weights = [val.cumprod() for val in self.test_conf.test_act_indiv_weights]


        test_act_traj_rew = [val.sum() for val in test_act_inidiv_rew]
        test_act_traj_weights = [val.prod() for val in self.test_conf.test_act_indiv_weights]
            
        test_act_traj_w_r = []
        for w,r in zip(test_act_traj_weights, test_act_traj_rew):
            test_act_traj_w_r.append(
                (
                    torch.Tensor([w]).squeeze(), 
                    torch.Tensor([r]).squeeze()
                    )
                )


        test_act_traj_w = [w for w,r in test_act_traj_w_r]

        test_act_losses = []
        for i,(w,r) in enumerate(test_act_traj_w_r):
            w = w/sum(test_act_traj_w)
            test_act_losses.append(w*r)
        test_act_loss = sum(test_act_losses).item()

        be_policy_mock = TestPolicy(self.test_conf.test_action_probs)
        behav_policy = MagicMock(
            spec=Policy,
            side_effect=be_policy_mock
            )
        #behav_policy.__call__ = MagicMock(side_effect=)
        #behav_policy = TestPolicy(self.test_conf.test_action_probs)
        self.is_sampler = ISWeightCalculator(behav_policy=behav_policy)
        # def __return_func(weight_array):
        #     return weight_array 
    
        # self.is_sampler.get_traj_weight_array = MagicMock(
        #     side_effect=__return_func)
        self.tollerance = [abs(val.mean())/1000 
                        for val in self.test_conf.test_act_indiv_weights]
        
        self.test_IS_weight_calculator = MagicMock(spec=ISWeightCalculator)
        self.test_IS_weight_calculator.is_weights = MagicMock(
            return_value=self.test_conf.weight_test_res)
        self.test_IS_weight_calculator.is_msk = MagicMock(
            return_value=self.test_conf.msk_test_res)

    
    def test_get_traj_w(self):
        test_pred = []
        #eval_policy = TestPolicy(self.test_conf.test_eval_action_probs)
        e_policy_mock = TestPolicy(self.test_conf.test_eval_action_probs)
        eval_policy = MagicMock(
            spec=Policy,
            side_effect=e_policy_mock
            )
        for s,a in zip(self.test_conf.test_state_vals, self.test_conf.test_action_vals):
            s = torch.Tensor(s)
            a = torch.Tensor(a)
            pred = self.is_sampler.get_traj_w(
                states=s, actions=a, eval_policy=eval_policy
            )
            self.assertEqual(pred.shape, torch.Size([s.shape[0]]))
            test_pred.append(pred.tolist())    
        for p,t,toll in zip(
            test_pred, 
            self.test_conf.test_act_indiv_weights, 
            self.tollerance):
            np.testing.assert_allclose(p, t, atol=toll)
    
    def test_get_dataset_w(self):
        input_states = [torch.Tensor(s) for s in self.test_conf.test_state_vals]
        input_actions = [torch.Tensor(a) for a in self.test_conf.test_action_vals]
        #eval_policy = TestPolicy(self.test_conf.test_eval_action_probs)
        e_policy_mock = TestPolicy(self.test_conf.test_eval_action_probs)
        eval_policy = MagicMock(
            spec=Policy,
            side_effect=e_policy_mock
            )
        is_weights, weight_msk = self.is_sampler.get_dataset_w(
            states=input_states, actions=input_actions, eval_policy=eval_policy)
        self.assertEqual(is_weights.shape, self.test_conf.weight_test_res.shape)
        self.assertEqual(weight_msk.shape, self.test_conf.weight_test_res.shape)
        tol = torch.Tensor(self.tollerance).view(-1,1).expand(
            size=(len(self.tollerance), is_weights.shape[1])).mean()
        np.testing.assert_allclose(
            is_weights.numpy(), self.test_conf.weight_test_res.numpy(), atol=tol.numpy()
            )
        np.testing.assert_allclose(
            weight_msk.numpy(), self.test_conf.msk_test_res.numpy(), atol=tol.numpy()
            )            
        
    # def test_eval_traj_reward(self):
        
    #     tollerance = abs(test_act_inidiv_rew.mean())/1000
    #     test_pred = []
    #     for r in test_reward_values:
    #         r = torch.Tensor(r)
    #         pred = self.is_sampler._ImportanceSampling__eval_traj_reward(
    #             reward_array=r
    #         )
    #         self.assertEqual(pred.shape, torch.Size([3]))
    #         test_pred.append(pred.tolist())
    #     test_pred = np.array(test_pred)
    #     res = test_pred==test_act_inidiv_rew
    #     if not res.all():
    #         logger.debug(test_pred)
    #         logger.debug(test_act_inidiv_rew)
    #         diff_res = test_pred-test_act_inidiv_rew
    #         diff_res = (diff_res < tollerance).all()
    #         self.assertTrue(diff_res)
    #     else:
    #         self.assertTrue(res.all())

@parameterized_class(test_configs_fmt_class)
class VanillaISTest(unittest.TestCase):
    
    test_conf:TestConfig
    
    def setUp(self) -> None:
        self.test_act_norm_conts = [val.prod() for val in self.test_conf.test_act_indiv_weights]
        self.test_IS_weight_calculator = MagicMock(spec=ISWeightCalculator)
        self.test_IS_weight_calculator.is_weights = MagicMock(
            return_value=self.test_conf.weight_test_res)
        self.test_IS_weight_calculator.is_msk = MagicMock(
            return_value=self.test_conf.msk_test_res)

        self.is_sampler = VanillaIS(is_weight_calc=self.test_IS_weight_calculator)

    def test_get_traj_weight_array(self):
        test_act_norm_conts_w_m = copy.deepcopy(self.test_conf.msk_test_res)
        for i in range(len(self.test_act_norm_conts)):
            test_act_norm_conts_w_m[i,:] = test_act_norm_conts_w_m[i,:]*self.test_act_norm_conts[i]

        tollerance_w_m = abs(test_act_norm_conts_w_m.numpy().mean())/1000
        test_act_norm_conts_w_m = torch.tensor(test_act_norm_conts_w_m)
        pred = self.is_sampler.get_traj_weight_array(
                is_weights=self.test_conf.weight_test_res, 
                weight_msk=self.test_conf.msk_test_res
            )
        
        self.assertEqual(pred.shape, test_act_norm_conts_w_m.shape)
        np.testing.assert_allclose(
            pred, test_act_norm_conts_w_m, atol=tollerance_w_m
        )                

@parameterized_class(test_configs_fmt_class)
class PerDecisionISTest(unittest.TestCase):
    
    test_conf:TestConfig
    
    def setUp(self) -> None:
        self.test_act_pd_weights = [val.cumprod() for val in self.test_conf.test_act_indiv_weights]
        self.test_IS_weight_calculator = MagicMock(spec=ISWeightCalculator)
        self.test_IS_weight_calculator.is_weights = MagicMock(
            return_value=self.test_conf.weight_test_res)
        self.test_IS_weight_calculator.is_msk = MagicMock(
            return_value=self.test_conf.msk_test_res)
        self.is_sampler = PerDecisionIS(is_weight_calc=self.test_IS_weight_calculator)

    def test_get_traj_weight_array(self):
        test_act_norm_conts_w_m = copy.deepcopy(self.test_conf.msk_test_res)
        for i in range(len(self.test_act_pd_weights)):
            test_act_norm_conts_w_m[i,0:len(self.test_act_pd_weights[i])] = torch.tensor(self.test_act_pd_weights[i])
            
        tollerance_w_m = abs(test_act_norm_conts_w_m.numpy().mean())/1000
        pred = self.is_sampler.get_traj_weight_array(
                is_weights=self.test_conf.weight_test_res, 
                weight_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred.shape, test_act_norm_conts_w_m.shape)
        np.testing.assert_allclose(
            pred, test_act_norm_conts_w_m, atol=tollerance_w_m
        )