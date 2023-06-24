import unittest
from unittest.mock import MagicMock
import torch
import logging
import numpy as np
import pandas as pd
# from dtr_renal.models.components.policy_eval.ImportanceSampling import (
#     ImportanceSampling, VanillaIS, PerDecisionIS)
# from dtr_renal.models.components.policy_eval.IsEvaluation import (
#     torch_is_evaluation)
from offline_rl_ope.components.ImportanceSampler import (
    ImportanceSampler, VanillaIS, PerDecisionIS, ISWeightCalculator)
from offline_rl_ope.Dataset import ISEpisode

logger = logging.getLogger("offline_rl_ope")

test_state_vals = [
    [[1,2,3,4], [5,6,7,8], [5,7,2,9], [5,7,2,9]],
    [[5,6,7,8], [5,6,7,8], [1,2,3,4]]
]
test_action_vals = [
    [[1], [0], [0], [1]],
    [[0], [0], [1]]
]

test_action_probs = [
    [[0.9], [0.7], [0.66], [0.7]],
    [[0.54], [0.9], [0.5]]
]

test_eval_action_vals = [
    [[1], [1], [0], [1]],
    [[0], [0], [0]]
]

test_eval_action_probs = [
    [[1], [0.07], [0.89], [1]],
    [[0.75], [0.9], [0.2]]
]

test_reward_values = [
    [[1],[-1], [1], [1]],
    [[-1],[-1], [-1]]
]

test_act_indiv_weights = [
    np.array([1/0.9, 0.07/0.7, 0.89/0.66, 1/0.7]),
    np.array([ 0.75/0.54, 0.9/0.9, 0.2/0.5])
    ]

test_act_inidiv_rew = [
    np.array([1, -1*0.99, 1*(np.power(0.99,2)), 1*(np.power(0.99,3))]),
    np.array([-1, -1*0.99, -1*(np.power(0.99,2))])
    ]

test_act_norm_conts = [val.prod() for val in test_act_indiv_weights]

test_act_pd_weights = [val.cumprod() for val in test_act_indiv_weights]


test_act_traj_rew = [val.sum() for val in test_act_inidiv_rew]
test_act_traj_weights = [val.prod() for val in test_act_indiv_weights]
       
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

# clip = 0.03
# test_act_losses_clip = []
# for i,(w,r) in enumerate(test_act_traj_w_r):
#     w = w/sum(test_act_traj_w)
#     if w > clip:
#         test_act_losses_clip.append(clip*r)
#     else:
#         test_act_losses_clip.append(w*r)
# test_act_loss_clip = sum(test_act_losses_clip).item()

class TestEvalPolicy:
    pass

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

class ISWeightCalculatorTest(unittest.TestCase):

    def setUp(self) -> None:
        behav_policy = TestPolicy(test_action_probs)
        eval_policy = TestPolicy(test_eval_action_probs)
        self.is_sampler = ISWeightCalculator(
            behav_policy=behav_policy, eval_policy=eval_policy)
        # def __return_func(weight_array):
        #     return weight_array 
    
        # self.is_sampler.get_traj_weight_array = MagicMock(
        #     side_effect=__return_func)
        self.tollerance = [abs(val.mean())/1000 
                           for val in test_act_indiv_weights]
    
    def test_get_traj_w(self):
        test_pred = []
        for s,a in zip(test_state_vals, test_action_vals):
            s = torch.Tensor(s)
            a = torch.Tensor(a)
            pred = self.is_sampler.get_traj_w(
                states=s, actions=a
            )
            self.assertEqual(pred.shape, torch.Size([s.shape[0]]))
            test_pred.append(pred.tolist())    
        for p,t,toll in zip(test_pred, test_act_indiv_weights, self.tollerance):
            res = (p==t).all()
            if not res:
                logger.debug("p: {}".format(p))
                logger.debug("t: {}".format(t))
                diff_res = p-t
                diff_res = (diff_res < toll).all()
                self.assertTrue(diff_res)
            else:
                self.assertTrue(res)
    
    def test_get_dataset_w(self):
        input_states = [torch.Tensor(s) for s in test_state_vals]
        input_actions = [torch.Tensor(a) for a in test_action_vals]
        pred_res = self.is_sampler.get_dataset_w(
            states=input_states, actions=input_actions, h=4)
        test_res = torch.Tensor(
            [
                test_act_indiv_weights[0].tolist(),
                [*test_act_indiv_weights[1].tolist(),1]
                ]
        )
        res = (pred_res == test_res).all()
        if not res:
            res_diff = pred_res - test_res
            tol = torch.Tensor(self.tollerance).view(-1,1).expand(
                size=(len(self.tollerance), res_diff.shape[1]))
            self.assertTrue((res_diff<tol).all())
        else:
            self.assertTrue(res)
            
        
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

class VanillaISTest(unittest.TestCase):
    def setUp(self) -> None:
        self.is_sampler = VanillaIS()

    def test_get_traj_weight_array(self):
        tollerance_w = [abs(val.mean())/1000 
                           for val in test_act_norm_conts]
        for i,w in enumerate(test_act_indiv_weights):
            w = torch.Tensor(w)
            pred = self.is_sampler.get_traj_weight_array(
                is_weights=w
            )
            #self.assertTrue(isinstance(pred, np.array))
            self.assertEqual(pred.shape, torch.Size([w.shape[0]]))
            res = pred==np.repeat(test_act_norm_conts[i],w.shape[0])
            if not res:
                logger.debug("pred: {}".format(pred))
                logger.debug("test_act_norm_conts[i]: {}".format(
                    test_act_norm_conts[i]))
                diff_res = pred-test_act_norm_conts[i]
                diff_res = (diff_res < tollerance_w[i]).all()
                self.assertTrue(diff_res)
            else:
                self.assertTrue(res)
                
                
class PerDecisionISTest(unittest.TestCase):
    def setUp(self) -> None:
        self.is_sampler = PerDecisionIS()

    def test_get_traj_weight_array(self):
        tollerance_w = [abs(val.mean())/1000 
                           for val in test_act_pd_weights]
        for i,w in enumerate(test_act_indiv_weights):
            w = torch.Tensor(w)
            pred = self.is_sampler.get_traj_weight_array(
                is_weights=w
            )
            #self.assertEqual(pred.shape, torch.Size([3]))
            res = pred==test_act_pd_weights[i]
            if not res:
                logger.debug("pred: {}".format(pred))
                logger.debug("test_act_pd_weights[i]: {}".format(
                    test_act_pd_weights[i]))
                diff_res = pred-test_act_pd_weights[i]
                diff_res = (diff_res < tollerance_w[i]).all()
                self.assertTrue(diff_res)
            else:
                self.assertTrue(res)

if __name__ == '__main__':
    unittest.main()