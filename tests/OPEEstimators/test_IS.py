import unittest
from unittest.mock import MagicMock
import torch
import numpy as np
from offline_rl_ope.OPEEstimators import (
    ISEstimator, 
    EmpiricalMeanDenom,
    PassWeightDenom
    )
from offline_rl_ope.api.StandardEstimators import (
    VanillaISPDIS, WeightedISPDIS
)
from torch.nn.functional import pad

from parameterized import parameterized
from ..base import test_configs_fmt, get_wpd_denoms


gamma = 0.99

class ISEstimatorTest(unittest.TestCase):
    
    def setUp(self) -> None:
        self.is_estimator = ISEstimator(
            weight_denom=PassWeightDenom(),
            empirical_denom=EmpiricalMeanDenom()
        )
    
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
            test_conf.weight_test_res.numpy()
            )
        test_res=test_res.sum(axis=1)
        #test_res = test_res.sum(axis=1).mean()
        tol = test_res.mean()/1000
        self.assertEqual(pred_res.shape, torch.Size((len(rewards),)))
        np.testing.assert_allclose(pred_res.numpy(), test_res, atol=tol)
    
    @parameterized.expand(test_configs_fmt)
    def test_predict_is1(self, name, test_conf):
        """
        .. math::
            \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{H-1} r_{n,t}\gamma^{t} \prod_{t=0}^{H-1} \frac{\pi_{e}(a_{t},s_{t})}{\pi_{\beta}(a_{t},s_{t})}
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
        is_estimator = VanillaISPDIS()
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

    # @parameterized.expand(test_configs_fmt)
    # def test_predict_is2(self, name, test_conf):
    #     """
    #     - https://arxiv.org/pdf/1906.03735 (snis when weights are IS)
    #     .. math::
    #         \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{H-1} r_{n,t}\gamma^{t} \prod_{t=0}^{H-1} \frac{\pi_{e}(a_{t},s_{t})}{\pi_{\beta}(a_{t},s_{t})}
    #     """
    #     traj_values = []
    #     rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
    #     weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
    #     states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
    #     actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
    #     for w, r in zip(
    #         weights,
    #         rewards
    #         ):
    #         np_pows = torch.arange(0, w.shape[0])
    #         np_discounts = torch.tensor([gamma]*w.shape[0])
    #         discnt_vals = torch.pow(np_discounts, np_pows)
    #         traj_values.append(
    #             torch.sum(
    #                 torch.multiply(
    #                     torch.multiply(r.squeeze(), discnt_vals),torch.prod(w)
    #                     ),
    #                 0,
    #                 keepdim=True
    #                 )
    #             )
    #     test_res = torch.concat(traj_values).sum()/len(traj_values)
    #     max_h = max([len(w) for w in weights])
    #     is_msk = torch.concat([
    #         torch.concat([torch.ones(x.shape[0]),torch.zeros(max_h-x.shape[0])])[None,:] 
    #         for x in weights
    #         ], axis=0)
    #     weight_tens = torch.concat([
    #         torch.concat([x, torch.ones(max_h-x.shape[0])])[None,:] for x in 
    #         weights
    #     ], axis=0)
    #     weight_tens = weight_tens.prod(1,keepdim=True).repeat(
    #         (1,is_msk.shape[1]))
    #     is_estimator = PDWIS()
    #     pred_res = is_estimator.predict(
    #         rewards=rewards,
    #         states=states,
    #         actions=actions,
    #         weights=weight_tens,
    #         discount=gamma,
    #         is_msk=is_msk
    #         )
    #     tol = np.abs(test_res.numpy().mean()/1000)
    #     np.testing.assert_allclose(
    #         pred_res.numpy(), 
    #         test_res.numpy(), 
    #         atol=tol
    #         )

    @parameterized.expand(test_configs_fmt)
    def test_predict_pd(self, name, test_conf):
        """
        .. math::
            \frac{1}{N} \sum_{n=1}^{N} \sum_{t=0}^{H-1} r_{n,t}\gamma^{t} \prod_{t'=0}^{t} \frac{\pi_{e}(a_{t'},s_{t'})}{\pi_{\beta}(a_{t'},s_{t'})}
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
        is_estimator = VanillaISPDIS()
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

    
        
    # @parameterized.expand(test_configs_fmt)
    # def test_predict_wis(self, name, test_conf):
    #     """
    #     .. math::
    #         w_{H}^{-1}\sum_{n=1}^{N} \sum_{t=1}^{H} r_{n,t}\gamma^{t-1} \prod_{t=1}^{H} \frac{\pi_{e}(a_{t},s_{t})}{\pi_{\beta}(a_{t},s_{t})}
    #         w_{H} = \sum_{n=1}^{N}\prod_{t=1}^{H} \frac{\pi_{e}(a_{t},s_{t})}{\pi_{\beta}(a_{t},s_{t})}
    #     """
    #     traj_values = []
    #     rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
    #     weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
    #     states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
    #     actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
    #     for w, r in zip(
    #         weights,
    #         rewards
    #         ):
    #         np_pows = torch.arange(0, w.shape[0])
    #         np_discounts = torch.tensor([gamma]*w.shape[0])
    #         discnt_vals = torch.pow(np_discounts, np_pows)
    #         traj_values.append(
    #             torch.sum(
    #                 torch.multiply(
    #                     torch.multiply(r.squeeze(), discnt_vals),torch.prod(w)
    #                     ),
    #                 0,
    #                 keepdim=True
    #                 )
    #             )
    #     denom = torch.concat(
    #         [torch.prod(w,0,keepdim=True) for w in weights]
    #         ).sum()
    #     test_res = torch.concat(traj_values).sum()/denom
    #     max_h = max([len(w) for w in weights])
    #     is_msk = torch.concat([
    #         torch.concat([torch.ones(x.shape[0]),torch.zeros(max_h-x.shape[0])])[None,:] 
    #         for x in weights
    #         ], axis=0)
    #     weight_tens = torch.concat([
    #         torch.concat([x, torch.ones(max_h-x.shape[0])])[None,:] for x in 
    #         weights
    #     ], axis=0)
    #     weight_tens = weight_tens.prod(1,keepdim=True).repeat(
    #         (1,is_msk.shape[1]))
    #     is_estimator = WeightedISPDIS()
    #     pred_res = is_estimator.predict(
    #         rewards=rewards,
    #         states=states,
    #         actions=actions,
    #         weights=weight_tens,
    #         discount=gamma,
    #         is_msk=is_msk
    #         )
    #     tol = np.abs(test_res.numpy().mean()/1000)
    #     np.testing.assert_allclose(
    #         pred_res.numpy(), 
    #         test_res.numpy(), 
    #         atol=tol
    #         )   

    # @parameterized.expand(test_configs_fmt)
    # def test_predict_wpd_precup(self, name, test_conf):
    #     """
    #     - http://incompleteideas.net/papers/PSS-00.pdf (Q^{PDW} when weights are PDW)
    #     .. math::
    #         w_{H}^{-1}\sum_{n=0}^{N} \sum_{t=0}^{H-1} r_{n,t}\gamma^{t} \prod_{t'=0}^{t} \frac{\pi_{e}(a_{t'},s_{t'})}{\pi_{\beta}(a_{t'},s_{t'})} \\
    #         w_{H} = \sum_{n=1}^{N}\sum_{t=0}^{H-1}\prod_{t=0}^{H-1} \frac{\pi_{e}(a_{t},s_{t})}{\pi_{\beta}(a_{t},s_{t})}
    #     """
    #     traj_values = []
    #     rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
    #     weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
    #     states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
    #     actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
    #     for w, r in zip(
    #         weights,
    #         rewards
    #         ):
    #         np_pows = torch.arange(0, w.shape[0])
    #         np_discounts = torch.tensor([gamma]*w.shape[0])
    #         discnt_vals = torch.pow(np_discounts, np_pows)
    #         traj_values.append(
    #             torch.sum(
    #                 torch.multiply(
    #                     torch.multiply(r.squeeze(), discnt_vals),torch.cumprod(w,0)
    #                     ),
    #                 0,
    #                 keepdim=True
    #                 )
    #             )
    #     denom = torch.concat(
    #         [torch.cumprod(w,0).sum(0,keepdim=True) for w in weights]
    #         ).sum()
    #     test_res = torch.concat(traj_values).sum()/denom
    #     max_h = max([len(w) for w in weights])
    #     is_msk = torch.concat([
    #         torch.concat([torch.ones(x.shape[0]),torch.zeros(max_h-x.shape[0])])[None,:] 
    #         for x in weights
    #         ], axis=0)
    #     weight_tens = torch.concat([
    #         torch.concat([x, torch.ones(max_h-x.shape[0])])[None,:] for x in 
    #         weights
    #     ], axis=0)
    #     weight_tens = weight_tens.cumprod(1)
    #     is_estimator = CumulativeVanillaPDWIS()
    #     pred_res = is_estimator.predict(
    #         rewards=rewards,
    #         states=states,
    #         actions=actions,
    #         weights=weight_tens,
    #         discount=gamma,
    #         is_msk=is_msk
    #         )
    #     tol = np.abs(test_res.numpy().mean()/1000)
    #     np.testing.assert_allclose(
    #         pred_res.numpy(), 
    #         test_res.numpy(), 
    #         atol=tol
    #         )

    @parameterized.expand(test_configs_fmt)
    def test_predict_pit_wis(self, name, test_conf):
        """
        - https://arxiv.org/pdf/1906.03735 (snsis when weights are IS)
        .. math::
            w_{H,i}=\prod_{t=0}^{H}w_{t,i}
            w_{H} = \frac{1}{n_{t}}\sum_{i=1}^{n_{t}} \mathbbm{1}w_{i,t}>0w_{H,i}
            n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{H,i}\frac{1}/{w_{H}}
        """
        traj_values = []
        rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
        weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
        states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
        actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
        h = max(w.shape[0] for w in weights)
        # w_{H} = \frac{1}{n_{t}}\sum_{i=1}^{n_{t}} \mathbbm{1}w_{i,t}>0w_{H,i}
        test_res_w = get_wpd_denoms(weights=weights,h=h,agg_type="mean",is_type="is")
        for w, r, wn in zip(
            weights,
            rewards,
            test_res_w
            ):
            np_pows = torch.arange(0, w.shape[0])
            np_discounts = torch.tensor([gamma]*w.shape[0])
            discnt_vals = torch.pow(np_discounts, np_pows)
            # w_{H,i}=\prod_{t=0}^{H}w_{t,i}
            prod_indiv_weights = torch.prod(w,0).repeat(w.shape[0])
            # w_{H,i}\frac{1}/{w_{H}}
            normed_weights = prod_indiv_weights/wn[0:prod_indiv_weights.shape[0]]
            # r_{t,i}\gamma^{t}w_{H,i}\frac{1}/{w_{H}}
            traj_values.append(
                    torch.multiply(
                        torch.multiply(r.squeeze(), discnt_vals), normed_weights
                        )
                    )
        traj_values = torch.concat(
            [pad(r, (0,h-r.shape[0]))[None,:] for r in traj_values]
            )
        # n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{H,i}\frac{1}/{w_{H}}
        test_res = (traj_values).sum()/len(weights)
        is_msk = torch.concat([
            torch.concat([torch.ones(x.shape[0]),torch.zeros(h-x.shape[0])])[None,:] 
            for x in weights
            ], axis=0)
        weight_tens = torch.concat([
            torch.concat([x, torch.ones(h-x.shape[0])])[None,:] for x in 
            weights
        ], axis=0)
        weight_tens = weight_tens.prod(1,keepdim=True).repeat(
            (1,is_msk.shape[1])
            )
        is_estimator = WeightedISPDIS()
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
        - https://arxiv.org/pdf/1906.03735 (snsis when weights are PD)
        .. math::
            w_{0:t',i}=\prod_{t=0}^{t'}w_{t,i}
            w_{0:t'} = \sum_{i=1}^{n_{t'}} \mathbbm{1}[w_{i,t}>0]w_{0:t',i}
            n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{0:t,i}\frac{1}/{n_{t}^{-1}w_{0:t}}
        """
        traj_values = []
        rewards = [torch.tensor(v).float() for v in test_conf.test_reward_values]
        weights = [torch.tensor(v).float() for v in test_conf.test_act_indiv_weights]
        states = [torch.tensor(v).float() for v in test_conf.test_state_vals]
        actions = [torch.tensor(v).float() for v in test_conf.test_action_vals]
        h = max(w.shape[0] for w in weights)
        # n_{t}^{-1}w_{0:t}
        test_res_w = get_wpd_denoms(weights=weights,h=h,agg_type="mean")
        for w, r, wn in zip(
            weights,
            rewards,
            test_res_w
            ):
            np_pows = torch.arange(0, w.shape[0])
            np_discounts = torch.tensor([gamma]*w.shape[0])
            discnt_vals = torch.pow(np_discounts, np_pows)
            # \prod_{t=0}^{t'}w_{t,i}
            cumprod_indiv_weights = torch.cumprod(w,0)
            # w_{0:t,i}\frac{1}/{n_{t}^{-1}w_{0:t}}
            normed_weights = cumprod_indiv_weights/wn[0:cumprod_indiv_weights.shape[0]]
            # r_{t,i}\gamma^{t}w_{0:t,i}\frac{1}/{n_{t}^{-1}w_{0:t}}
            traj_values.append(
                    torch.multiply(
                        torch.multiply(r.squeeze(), discnt_vals), normed_weights
                        )
                    )
        traj_values = torch.concat(
            [pad(r, (0,h-r.shape[0]))[None,:] for r in traj_values]
            )
        # n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{0:t,i}\frac{1}/{n_{t}^{-1}w_{0:t}}
        test_res = (traj_values).sum()/len(weights)
        is_msk = torch.concat([
            torch.concat([torch.ones(x.shape[0]),torch.zeros(h-x.shape[0])])[None,:] 
            for x in weights
            ], axis=0)
        weight_tens = torch.concat([
            torch.concat([x, torch.ones(h-x.shape[0])])[None,:] for x in 
            weights
        ], axis=0)
        weight_tens = weight_tens.cumprod(1)
        is_estimator = WeightedISPDIS()
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
