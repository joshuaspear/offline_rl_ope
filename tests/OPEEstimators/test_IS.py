import unittest
from unittest.mock import MagicMock
import torch
import numpy as np
from offline_rl_ope.OPEEstimators.IS import ISEstimator
from parameterized import parameterized
from ..base import test_configs_fmt


gamma = 0.99

class ISEstimatorTest(unittest.TestCase):
    
    def setUp(self) -> None:
        self.is_estimator = ISEstimator(norm_weights=False)
    
    @parameterized.expand(test_configs_fmt)
    def test_get_traj_discnt_reward(self, name, test_conf):
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
    
    @parameterized.expand(test_configs_fmt)
    def test_get_dataset_discnt_reward(self, name, test_conf):
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
            rewards=rewards, discount=gamma, 
            h=test_conf.reward_test_res.shape[1]
            )
        self.assertTrue(pred_res.shape, test_conf.reward_test_res.shape)
        np.testing.assert_allclose(pred_res.numpy(),test_conf.reward_test_res.numpy(),
                                np.abs(test_conf.reward_test_res.mean().numpy()))
        
    @parameterized.expand(test_configs_fmt)
    def test_predict_traj_rewards(self, name, test_conf):
        def __mock_return(rewards, discount, h):
            return test_conf.reward_test_res
        self.is_estimator.get_dataset_discnt_reward = MagicMock(
            side_effect=__mock_return)
        rewards = [torch.Tensor(r) for r in test_conf.test_reward_values]
        pred_res = self.is_estimator.predict_traj_rewards(
            rewards=rewards, actions=[], states=[], 
            weights=test_conf.weight_test_res,
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
    
    @parameterized.expand(test_configs_fmt)
    def test_predict_is(self, name, test_conf):
        """
        .. math::
            \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{H} r_{n,t}\gamma^{t-1} \prod_{t=1}^{H} \frac{\pi_{e}(a_{t},s_{t})}{\pi_{\beta}(a_{t},s_{t})}
        """
        traj_values = []
        rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
        weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
        states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
        actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
        for w, r in zip(
            weights,
            rewards
            ):
            np_pows = torch.arange(0, w.shape[0])
            np_discounts = torch.tensor([gamma]*w.shape[0])
            discnt_vals = torch.pow(np_discounts, np_pows)
            traj_values.append(
                torch.sum(
                    torch.multiply(
                        torch.multiply(r.squeeze(), discnt_vals),torch.prod(w)
                        ),
                    0,
                    keepdim=True
                    )
                )
        test_res = torch.concat(traj_values).sum()/len(traj_values)
        max_h = max([len(w) for w in weights])
        is_msk = torch.concat([
            torch.concat([torch.ones(x.shape[0]),torch.zeros(max_h-x.shape[0])])[None,:] 
            for x in weights
            ], axis=0)
        weight_tens = torch.concat([
            torch.concat([x, torch.ones(max_h-x.shape[0])])[None,:] for x in 
            weights
        ], axis=0)
        weight_tens = weight_tens.prod(1,keepdim=True).repeat(
            (1,is_msk.shape[1]))
        is_estimator = ISEstimator(
            norm_weights=False
            )
        pred_res = is_estimator.predict(
            rewards=rewards,
            states=states,
            actions=actions,
            weights=weight_tens,
            discount=gamma,
            is_msk=is_msk
            )
        tol = np.abs(test_res.numpy().mean()/1000)
        np.testing.assert_allclose(
            pred_res.numpy(), 
            test_res.numpy(), 
            atol=tol
            )

    @parameterized.expand(test_configs_fmt)
    def test_predict_pd(self, name, test_conf):
        """
        .. math::
            \frac{1}{N} \sum_{n=1}^{N} \sum_{t=1}^{H} r_{n,t}\gamma^{t-1} \prod_{t'=1}^{t} \frac{\pi_{e}(a_{t'},s_{t'})}{\pi_{\beta}(a_{t'},s_{t'})}
        """
        traj_values = []
        rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
        weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
        states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
        actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
        for w, r in zip(
            weights,
            rewards
            ):
            np_pows = torch.arange(0, w.shape[0])
            np_discounts = torch.tensor([gamma]*w.shape[0])
            discnt_vals = torch.pow(np_discounts, np_pows)
            traj_values.append(
                torch.sum(
                    torch.multiply(
                        torch.multiply(r.squeeze(), discnt_vals),torch.cumprod(w,0)
                        ),
                    0,
                    keepdim=True
                    )
                )
        test_res = torch.concat(traj_values).sum()/len(traj_values)
        max_h = max([len(w) for w in weights])
        is_msk = torch.concat([
            torch.concat([torch.ones(x.shape[0]),torch.zeros(max_h-x.shape[0])])[None,:] 
            for x in weights
            ], axis=0)
        weight_tens = torch.concat([
            torch.concat([x, torch.ones(max_h-x.shape[0])])[None,:] for x in 
            weights
        ], axis=0)
        weight_tens = weight_tens.cumprod(1)
        is_estimator = ISEstimator(
            norm_weights=False
            )
        pred_res = is_estimator.predict(
            rewards=rewards,
            states=states,
            actions=actions,
            weights=weight_tens,
            discount=gamma,
            is_msk=is_msk
            )
        tol = np.abs(test_res.numpy().mean()/1000)
        np.testing.assert_allclose(
            pred_res.numpy(), 
            test_res.numpy(), 
            atol=tol
            )

    @parameterized.expand(test_configs_fmt)
    def test_predict_wis(self, name, test_conf):
        """
        .. math::
            w_{H}^{-1}\sum_{n=1}^{N} \sum_{t=1}^{H} r_{n,t}\gamma^{t-1} \prod_{t=1}^{H} \frac{\pi_{e}(a_{t},s_{t})}{\pi_{\beta}(a_{t},s_{t})}
            w_{H} = \sum_{n=1}^{N}\prod_{t=1}^{H} \frac{\pi_{e}(a_{t},s_{t})}{\pi_{\beta}(a_{t},s_{t})}
        """
        traj_values = []
        rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
        weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
        states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
        actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
        for w, r in zip(
            weights,
            rewards
            ):
            np_pows = torch.arange(0, w.shape[0])
            np_discounts = torch.tensor([gamma]*w.shape[0])
            discnt_vals = torch.pow(np_discounts, np_pows)
            traj_values.append(
                torch.sum(
                    torch.multiply(
                        torch.multiply(r.squeeze(), discnt_vals),torch.prod(w)
                        ),
                    0,
                    keepdim=True
                    )
                )
        denom = torch.concat(
            [torch.prod(w,0,keepdim=True) for w in weights]
            ).sum()
        test_res = torch.concat(traj_values).sum()/denom
        max_h = max([len(w) for w in weights])
        is_msk = torch.concat([
            torch.concat([torch.ones(x.shape[0]),torch.zeros(max_h-x.shape[0])])[None,:] 
            for x in weights
            ], axis=0)
        weight_tens = torch.concat([
            torch.concat([x, torch.ones(max_h-x.shape[0])])[None,:] for x in 
            weights
        ], axis=0)
        weight_tens = weight_tens.prod(1,keepdim=True).repeat(
            (1,is_msk.shape[1]))
        is_estimator = ISEstimator(
            norm_weights=True
            )
        pred_res = is_estimator.predict(
            rewards=rewards,
            states=states,
            actions=actions,
            weights=weight_tens,
            discount=gamma,
            is_msk=is_msk
            )
        tol = np.abs(test_res.numpy().mean()/1000)
        np.testing.assert_allclose(
            pred_res.numpy(), 
            test_res.numpy(), 
            atol=tol
            )

    @parameterized.expand(test_configs_fmt)
    def test_predict_wpd(self, name, test_conf):
        """
        .. math::
            w_{H}^{-1}\sum_{n=1}^{N} \sum_{t=1}^{H} r_{n,t}\gamma^{t-1} \prod_{t=1}^{H} \frac{\pi_{e}(a_{t},s_{t})}{\pi_{\beta}(a_{t},s_{t})}
            w_{H} = \sum_{n=1}^{N}\prod_{t=1}^{H} \frac{\pi_{e}(a_{t},s_{t})}{\pi_{\beta}(a_{t},s_{t})}
        """
        traj_values = []
        rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
        weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
        states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
        actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
        for w, r in zip(
            weights,
            rewards
            ):
            np_pows = torch.arange(0, w.shape[0])
            np_discounts = torch.tensor([gamma]*w.shape[0])
            discnt_vals = torch.pow(np_discounts, np_pows)
            traj_values.append(
                torch.sum(
                    torch.multiply(
                        torch.multiply(r.squeeze(), discnt_vals),torch.cumprod(w,0)
                        ),
                    0,
                    keepdim=True
                    )
                )
        denom = torch.concat(
            [torch.cumprod(w,0).sum(0,keepdim=True) for w in weights]
            ).sum()
        test_res = torch.concat(traj_values).sum()/denom
        max_h = max([len(w) for w in weights])
        is_msk = torch.concat([
            torch.concat([torch.ones(x.shape[0]),torch.zeros(max_h-x.shape[0])])[None,:] 
            for x in weights
            ], axis=0)
        weight_tens = torch.concat([
            torch.concat([x, torch.ones(max_h-x.shape[0])])[None,:] for x in 
            weights
        ], axis=0)
        weight_tens = weight_tens.cumprod(1)
        is_estimator = ISEstimator(
            norm_weights=True,
            norm_kwargs={"cumulative":True}
            )
        pred_res = is_estimator.predict(
            rewards=rewards,
            states=states,
            actions=actions,
            weights=weight_tens,
            discount=gamma,
            is_msk=is_msk
            )
        tol = np.abs(test_res.numpy().mean()/1000)
        np.testing.assert_allclose(
            pred_res.numpy(), 
            test_res.numpy(), 
            atol=tol
            )
