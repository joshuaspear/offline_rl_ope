import unittest
import torch
import logging
import numpy as np
import pandas as pd
# from dtr_renal.models.components.policy_eval.ImportanceSampling import (
#     ImportanceSampling, VanillaIS, PerDecisionIS)
# from dtr_renal.models.components.policy_eval.IsEvaluation import (
#     torch_is_evaluation)
from offline_rl_ope.components.ImportanceSampling import (
    ImportanceSampling, VanillaIS, PerDecisionIS)

logger = logging.getLogger("offline_rl_ope")

test_state_vals = [
    [[1,2,3,4], [5,6,7,8], [5,7,2,9]],
    [[5,6,7,8], [5,6,7,8], [1,2,3,4]]
]
test_action_vals = [
    [[1], [0], [0]],
    [[0], [0], [1]]
]

test_action_probs = [
    [[0.9], [0.7], [0.66]],
    [[0.54], [0.9], [0.5]]
]

test_eval_action_vals = [
    [[1], [1], [0]],
    [[0], [0], [0]]
]

test_eval_action_probs = [
    [[1], [0.07], [0.89]],
    [[0.75], [0.9], [0.2]]
]

test_reward_values = [
    [[1],[-1], [1]],
    [[-1],[-1], [-1]]
]

test_act_indiv_weights = np.array([
    [1/0.9, 0.07/0.7, 0.89/0.66],
    [ 0.75/0.54, 0.9/0.9, 0.2/0.5]
    ])

test_act_inidiv_rew = np.array(
    [[1, -1*0.99, 1*(np.power(0.99,2))],
     [-1, -1*0.99, -1*(np.power(0.99,2))]]
    )


test_act_traj_rew = test_act_inidiv_rew.sum(axis=1)
test_act_traj_weights = test_act_indiv_weights.prod(axis=1)        
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

class ImportanceSamplingTest(unittest.TestCase):

    def setUp(self) -> None:
        behav_policy = TestPolicy(test_action_probs)
        eval_policy = TestPolicy(test_eval_action_probs)
        self.is_sampler = ImportanceSampling(
            behav_policy=behav_policy, eval_policy=eval_policy, 
            discount=0.99)
    
    def test_eval_array_weight(self):
        tollerance = abs(test_act_indiv_weights.mean())/1000
        test_pred = []
        for s,a in zip(test_state_vals, test_action_vals):
            s = torch.Tensor(s)
            a = torch.Tensor(a)
            pred = self.is_sampler._ImportanceSampling__eval_array_weight(
                state_array=s, action_array=a
            )
            self.assertEqual(pred.shape, torch.Size([3]))
            test_pred.append(pred.tolist())
        test_pred = np.array(test_pred)
        res = test_pred==test_act_indiv_weights
        if not res.all():
            logger.debug(test_pred)
            logger.debug(test_act_indiv_weights)
            diff_res = test_pred-test_act_indiv_weights
            diff_res = (diff_res < tollerance).all()
            self.assertTrue(diff_res)
        else:
            self.assertTrue(res.all())
            

    def test_eval_traj_reward(self):
        
        tollerance = abs(test_act_inidiv_rew.mean())/1000
        test_pred = []
        for r in test_reward_values:
            r = torch.Tensor(r)
            pred = self.is_sampler._ImportanceSampling__eval_traj_reward(
                reward_array=r
            )
            self.assertEqual(pred.shape, torch.Size([3]))
            test_pred.append(pred.tolist())
        test_pred = np.array(test_pred)
        res = test_pred==test_act_inidiv_rew
        if not res.all():
            logger.debug(test_pred)
            logger.debug(test_act_inidiv_rew)
            diff_res = test_pred-test_act_inidiv_rew
            diff_res = (diff_res < tollerance).all()
            self.assertTrue(diff_res)
        else:
            self.assertTrue(res.all())

class VanillaISTest(unittest.TestCase):
    def setUp(self) -> None:
        behav_policy = TestPolicy(test_action_probs)
        eval_policy = TestPolicy(test_eval_action_probs)
        self.is_sampler = VanillaIS(
            behav_policy=behav_policy, eval_policy=eval_policy, 
            discount=0.99)
    
    def test_get_traj_rwrd(self):
        
        tollerance_w = abs(test_act_traj_weights.mean())/1000
        tollerance_r = abs(test_act_traj_rew.mean())/1000
        for i,(w,r) in enumerate(
            zip(test_act_indiv_weights, test_act_inidiv_rew)):
            w = torch.Tensor(w)
            r = torch.Tensor(r)
            pred = self.is_sampler._ImportanceSampling__get_traj_rwrd(
                weight_array=w, discnt_reward_array=r
            )
            self.assertTrue(isinstance(pred, tuple))
            self.assertTrue(len(pred)==2)
            self.assertEqual(pred[0].shape, torch.Size([]))
            self.assertEqual(pred[1].shape, torch.Size([]))
            res = pred==test_act_traj_w_r[i]
            if not res:
                logger.debug("w: {}".format(w))
                logger.debug("r: {}".format(r))
                logger.debug("pred: {}".format(pred))
                logger.debug("test_act[i]: {}".format(test_act_traj_w_r[i]))
                diff_res = pred[0]-test_act_traj_w_r[i][0]
                diff_res = (diff_res < tollerance_w).all()
                self.assertTrue(diff_res)
                diff_res = pred[1]-test_act_traj_w_r[i][1]
                diff_res = (diff_res < tollerance_r).all()
                self.assertTrue(diff_res)
            else:
                self.assertTrue(res)

# class PerDecisionISTest(unittest.TestCase):
#     def setUp(self) -> None:
#         behav_policy = TestPolicy(test_action_probs)
#         eval_policy = TestPolicy(test_eval_action_probs)
#         self.is_sampler = VanillaIS(
#             behav_policy=behav_policy, eval_policy=eval_policy, 
#             discount=0.99)
    
#     def test_get_traj_rwrd(self):
#         test_act_traj_rew = test_act_inidiv_rew.copy()
#         test_act_traj_weights = test_act_indiv_weights.cumprod(axis=1)
#         test_act = []
#         for w,r in zip(test_act_traj_weights, test_act_traj_rew):
#             test_act.append(
#                 (
#                     torch.Tensor([w]).squeeze(), 
#                     torch.Tensor([r]).squeeze()
#                     )
#                 )
#         tollerance_w = abs(test_act_traj_weights.mean())/1000
#         tollerance_r = abs(test_act_traj_rew.mean())/1000
#         for i,(w,r) in enumerate(
#             zip(test_act_indiv_weights, test_act_inidiv_rew)):
#             w = torch.Tensor(w)
#             r = torch.Tensor(r)
#             pred = self.is_sampler._ImportanceSampling__get_traj_rwrd(
#                 weight_array=w, discnt_reward_array=r
#             )
#             self.assertTrue(isinstance(pred, tuple))
#             self.assertTrue(len(pred)==2)
#             self.assertEqual(pred[0].shape, torch.Size([]))
#             self.assertEqual(pred[1].shape, torch.Size([]))
#             res = pred==test_act[i]
#             if not res.all():
#                 logger.debug("w: {}".format(w))
#                 logger.debug("r: {}".format(r))
#                 logger.debug("pred: {}".format(pred))
#                 logger.debug("test_act[i]: {}".format(test_act[i]))
#                 diff_res = pred[0]-test_act[i][0]
#                 diff_res = (diff_res < tollerance_w).all()
#                 self.assertTrue(diff_res)
#                 diff_res = pred[1]-test_act[i][1]
#                 diff_res = (diff_res < tollerance_r).all()
#                 self.assertTrue(diff_res)
#             else:
#                 self.assertTrue(res.all())


# class torch_is_evaluation_Test(unittest.TestCase):
    
#     def test_torch_is_evaluation_vanillais(self):
#         behav_policy = TestPolicy(test_action_probs)
#         eval_policy = TestPolicy(test_eval_action_probs)
#         is_est = VanillaIS(
#             behav_policy=behav_policy, eval_policy=eval_policy, 
#             discount=0.99)
#         dataset = []
#         for s,a,r in zip(test_state_vals, test_action_vals, test_reward_values): 
#             state_data = torch.Tensor(s)
#             action_data = torch.Tensor(a)
#             reward_data = torch.Tensor(r)
#             dataset.append(
#                 {
#                     "state": state_data, 
#                     "act": action_data, 
#                     "reward": reward_data
#                     }
#                 )
#         res = torch_is_evaluation(
#             importance_sampler=is_est, dataset=dataset, 
#             norm_weights=True, save_dir=None, 
#             prefix=None, clip=None)
#         logger.debug(res)
#         act = (
#             test_act_loss, 
#             torch.Tensor(test_act_losses), 
#             test_act_loss, 
#             torch.Tensor(test_act_losses)
#             # test_act_loss_clip, 
#             # test_act_losses_clip
#             )
#         logger.debug(act)
#         for i,j in zip(res, act):
#             is_tense = isinstance(j, torch.Tensor)
#             if is_tense:
#                 tollerance = (j.mean()).item()/1000
#             else:
#                 tollerance = j/1000
#             if is_tense:
#                 res = torch.eq(i,j).all().item()
#             else:
#                 res = i==j
#             if not res:
#                 diff_res = i-j
#                 diff_res = (abs(diff_res) < abs(tollerance))
#                 if is_tense:
#                     diff_res = diff_res.all()
#                 self.assertTrue(diff_res)
#             else:
#                 self.assertTrue(res)

if __name__ == '__main__':
    unittest.main()