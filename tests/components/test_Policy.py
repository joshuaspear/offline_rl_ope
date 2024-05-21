import unittest
from unittest.mock import MagicMock
import torch
import numpy as np
from offline_rl_ope.components.Policy import (
    GreedyDeterministic, BehavPolicy)
from offline_rl_ope import logger
# from ..base import (test_action_probs, test_action_vals, test_eval_action_probs, 
#                     test_eval_action_vals, test_reward_values, test_state_vals)
from ..base import (
    single_discrete_action_test as sdat,
    duel_discrete_action_test as ddat,
    bin_discrete_action_test as bdat
    )

eps = 0.001

for test_conf in [sdat,ddat,bdat]:
    class GreedyDeterministicTest(unittest.TestCase):

        def setUp(self) -> None:
            def __mock_return(x):
                lkp = {
                    str(torch.Tensor(state)):torch.Tensor(act) 
                    for state,act in zip(
                        test_conf.test_state_vals, 
                        test_conf.test_eval_action_vals
                        )
                    }
                return lkp[str(x)]
            policy_func = MagicMock(side_effect=__mock_return)
            self.policy_0_eps = GreedyDeterministic(policy_func, gpu=False)
            self.policy_001_eps = GreedyDeterministic(
                policy_func, gpu=False, eps=eps)
            
            def __mock_return_multi_dim(x):
                lkp = {
                    str(torch.Tensor(state)):torch.concat(
                        [torch.Tensor(act),torch.abs(1-torch.Tensor(act))],
                        dim=1
                        ) 
                    for state,act in zip(
                        test_conf.test_state_vals, 
                        test_conf.test_eval_action_vals
                        )
                    }
                return lkp[str(x)]
            policy_func_multi_dim = MagicMock(side_effect=__mock_return_multi_dim)
            self.policy_0_eps_multi_dim = GreedyDeterministic(
                policy_func_multi_dim, 
                gpu=False
                )
            self.policy_001_eps_multi_dim = GreedyDeterministic(
                policy_func_multi_dim, 
                gpu=False, 
                eps=eps
                )
        
        def test___call__0_eps(self):
            test_pred = []
            __test_action_vals = [np.array(i) for i in test_conf.test_action_vals]
            __test_eval_action_vals = [np.array(i) for i in test_conf.test_eval_action_vals]
            test_res = [(x==y).astype(int) 
                for x,y in zip(__test_action_vals, __test_eval_action_vals)]
            test_res = np.concatenate(test_res).squeeze()
            tollerance = test_res.mean()/1000    
            for s,a in zip(test_conf.test_state_vals, __test_action_vals):
                s = torch.Tensor(s)
                a = torch.Tensor(a)
                assert len(s.shape) == 2, "Incorrect test input dimensions"
                assert len(a.shape) == 2, "Incorrect test input dimensions"
                pred = self.policy_0_eps(state=s, action=a)
                self.assertEqual(pred.shape, torch.Size((s.shape[0],1)))
                test_pred.append(pred.squeeze().numpy())
            test_pred = np.concatenate(test_pred)
            np.testing.assert_allclose(test_pred, test_res, atol=tollerance)
            
        def test___call__0001_eps(self):
            test_pred = []
            __test_action_vals = [np.array(i) for i in test_conf.test_action_vals]
            __test_eval_action_vals = [np.array(i) for i in test_conf.test_eval_action_vals]
            test_res = [(x==y).astype(int) 
                for x,y in zip(__test_action_vals, __test_eval_action_vals)]
            test_res = np.concatenate(test_res).squeeze()
            test_res = np.where(
                test_res == 1, 1-eps, 0+eps
            )
            tollerance = test_res.mean()/1000    
            for s,a in zip(test_conf.test_state_vals, __test_action_vals):
                s = torch.Tensor(s)
                a = torch.Tensor(a)
                assert len(s.shape) == 2, "Incorrect test input dimensions"
                assert len(a.shape) == 2, "Incorrect test input dimensions"
                pred = self.policy_001_eps(state=s, action=a)
                self.assertEqual(pred.shape, torch.Size((s.shape[0],1)))
                test_pred.append(pred.squeeze().numpy())
            test_pred = np.concatenate(test_pred)
            np.testing.assert_allclose(test_pred, test_res, atol=tollerance)

        # def test___call__0_eps_multi_dim(self):
        #         test_pred = []
        #         __test_action_vals = [
        #             np.concatenate(
        #                 [np.array(i),np.abs(1-np.array(i))],
        #                 axis=1
        #                 ) for i in test_conf.test_action_vals
        #             ]
        #         __test_eval_action_vals = [
        #             np.concatenate(
        #                 [np.array(i),np.abs(1-np.array(i))],
        #                 axis=1
        #                 ) for i in test_conf.test_eval_action_vals
        #             ]
        #         test_res = [(x==y).all(axis=1).astype(int) 
        #             for x,y in zip(__test_action_vals, __test_eval_action_vals)]
        #         test_res = np.concatenate(test_res).squeeze()
        #         tollerance = test_res.mean()/1000    
        #         for s,a in zip(test_conf.test_state_vals, __test_action_vals):
        #             s = torch.Tensor(s)
        #             a = torch.Tensor(a)
        #             pred = self.policy_0_eps_multi_dim(state=s, action=a)
        #             self.assertEqual(pred.shape, torch.Size((s.shape[0],1)))
        #             test_pred.append(pred.squeeze().numpy())
        #         test_pred = np.concatenate(test_pred)
        #         np.testing.assert_allclose(test_pred, test_res, atol=tollerance)
                
        # def test___call__0001_eps_multi_dim(self):
        #     test_pred = []
        #     __test_action_vals = [
        #         np.concatenate(
        #             [np.array(i),np.abs(1-np.array(i))],
        #             axis=1
        #             ) for i in test_conf.test_action_vals
        #         ]
        #     __test_eval_action_vals = [
        #         np.concatenate(
        #             [np.array(i),np.abs(1-np.array(i))],
        #             axis=1
        #             ) for i in test_conf.test_eval_action_vals
        #         ]
        #     test_res = [(x==y).all(axis=1).astype(int) 
        #         for x,y in zip(__test_action_vals, __test_eval_action_vals)]
        #     test_res = np.concatenate(test_res).squeeze()
        #     test_res = np.where(
        #         test_res == 1, 1-eps, 0+eps
        #     )
        #     tollerance = test_res.mean()/1000    
        #     for s,a in zip(test_conf.test_state_vals, __test_action_vals):
        #         s = torch.Tensor(s)
        #         a = torch.Tensor(a)
        #         pred = self.policy_001_eps_multi_dim(state=s, action=a)
        #         self.assertEqual(pred.shape, torch.Size((s.shape[0],1)))
        #         test_pred.append(pred.squeeze().numpy())
        #     test_pred = np.concatenate(test_pred)
        #     np.testing.assert_allclose(test_pred, test_res, atol=tollerance)

    class MockPolicyClass:
        
        def __init__(self) -> None:
            pass
            
    class BehavPolicyTest(unittest.TestCase):
        
        def setUp(self) -> None:
            def __mock_return(y, x):
                lkp = {
                    "_".join(
                        [
                            str(torch.tensor(state).float()), 
                            str(torch.tensor(act).float())
                            ]
                        ): torch.tensor(probs) 
                    for state,act,probs in zip(
                        test_conf.test_state_vals, test_conf.test_action_vals, 
                        test_conf.test_action_probs)
                    }
                print(f"x: {x}")
                print(f"y: {y}")
                print(f"lkp: {list(lkp.keys())[0]}")
                print(f'id: {"_".join([str(x),str(y)])}')
                return lkp["_".join([str(x),str(y)])]
            #policy_func = MockPolicyClass()
            #policy_func.__call__ = MagicMock(side_effect=__mock_return)
            #self.policy = BehavPolicy(policy_func)
            self.policy = BehavPolicy(
                policy_func=MagicMock(side_effect=__mock_return))

        
        def test___call__(self):
            test_pred = []
            test_res = [np.array(i) for i in test_conf.test_action_probs]
            test_res = np.concatenate(test_res).squeeze()
            tollerance = test_res.mean()/1000
            for s,a in zip(test_conf.test_state_vals, test_conf.test_action_vals):
                s = torch.Tensor(s)
                a = torch.Tensor(a)
                assert len(s.shape) == 2, "Incorrect test input dimensions"
                assert len(a.shape) == 2, "Incorrect test input dimensions"
                pred = self.policy(state=s, action=a)
                self.assertEqual(pred.shape, torch.Size((s.shape[0],1)))
                test_pred.append(pred.squeeze().numpy())
            test_pred = np.concatenate(test_pred)
            np.testing.assert_allclose(test_pred, test_res, atol=tollerance)

                
