import unittest
from unittest.mock import MagicMock
import torch
import logging
import numpy as np
from offline_rl_ope.components.Policy import (
    D3RlPyDeterministic, BehavPolicy)
from ..base import (test_action_probs, test_action_vals, test_eval_action_probs, 
                    test_eval_action_vals, test_reward_values, test_state_vals)


logger = logging.getLogger("offline_rl_ope")


eps = 0.001

class D3RlPyDeterministicTest(unittest.TestCase):

    def setUp(self) -> None:
        def __mock_return(x):
            lkp = {
                str(torch.Tensor(state)):torch.Tensor(act) 
                for state,act in zip(test_state_vals, test_eval_action_vals)
                }
            return lkp[str(x)]
        policy_class = MagicMock(side_effect=__mock_return)
        self.policy_0_eps = D3RlPyDeterministic(policy_class, gpu=False)
        self.policy_001_eps = D3RlPyDeterministic(
            policy_class, gpu=False, eps=eps)
    
    def test___call__0_eps(self):
        test_pred = []
        __test_action_vals = [np.array(i) for i in test_action_vals]
        __test_eval_action_vals = [np.array(i) for i in test_eval_action_vals]
        test_res = [(x==y).astype(int) 
               for x,y in zip(__test_action_vals, __test_eval_action_vals)]
        test_res = np.concatenate(test_res).squeeze()
        tollerance = test_res.mean()/1000    
        for s,a in zip(test_state_vals, test_action_vals):
            s = torch.Tensor(s)
            a = torch.Tensor(a)
            pred = self.policy_0_eps(state=s, action=a)
            self.assertEqual(pred.shape, torch.Size((s.shape[0],1)))
            test_pred.append(pred.squeeze().numpy())
        test_pred = np.concatenate(test_pred)
        np.testing.assert_allclose(test_pred, test_res, atol=tollerance)
        
    def test___call__0001_eps(self):
        test_pred = []
        __test_action_vals = [np.array(i) for i in test_action_vals]
        __test_eval_action_vals = [np.array(i) for i in test_eval_action_vals]
        test_res = [(x==y).astype(int) 
               for x,y in zip(__test_action_vals, __test_eval_action_vals)]
        test_res = np.concatenate(test_res).squeeze()
        test_res = np.where(
            test_res == 1, 1-eps, 0+eps
        )
        tollerance = test_res.mean()/1000    
        for s,a in zip(test_state_vals, test_action_vals):
            s = torch.Tensor(s)
            a = torch.Tensor(a)
            pred = self.policy_001_eps(state=s, action=a)
            self.assertEqual(pred.shape, torch.Size((s.shape[0],1)))
            test_pred.append(pred.squeeze().numpy())
        test_pred = np.concatenate(test_pred)
        np.testing.assert_allclose(test_pred, test_res, atol=tollerance)

class MockPolicyClass:
    
    def __init__(self) -> None:
        pass
        
class BehavPolicyTest(unittest.TestCase):
    
    def setUp(self) -> None:
        def __mock_return(dep_vals, indep_vals):
            lkp = {
                "_".join([str(np.array(state).astype(float)),
                          str(np.array(act).astype(float))]): np.array(probs) 
                for state,act,probs in zip(
                    test_state_vals, test_action_vals, 
                    test_action_probs)
                }
            return lkp["_".join([str(indep_vals),str(dep_vals)])]
        policy_class = MockPolicyClass()
        policy_class.eval_pdf = MagicMock(side_effect=__mock_return)
        self.policy = BehavPolicy(policy_class)

    
    def test___call__(self):
        test_pred = []
        test_res = [np.array(i) for i in test_action_probs]
        test_res = np.concatenate(test_res).squeeze()
        tollerance = test_res.mean()/1000
        for s,a in zip(test_state_vals, test_action_vals):
            s = torch.Tensor(s)
            a = torch.Tensor(a)
            pred = self.policy(state=s, action=a)
            self.assertEqual(pred.shape, torch.Size((s.shape[0],1)))
            test_pred.append(pred.squeeze().numpy())
        test_pred = np.concatenate(test_pred)
        np.testing.assert_allclose(test_pred, test_res, atol=tollerance)

            
