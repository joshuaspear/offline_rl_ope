import copy
import numpy as np
import torch
from torch.nn.functional import pad
import unittest
from offline_rl_ope.OPEEstimators import (
    PassWeightDenom, AvgWeightDenom, AvgPiTWeightDenom, 
    SumPiTWeightDenom
)
from parameterized import parameterized_class
from ..base import test_configs_fmt_class, TestConfig, get_wpd_denoms

@parameterized_class(test_configs_fmt_class)
class UtilsTestWeightDenom(unittest.TestCase):
    
    test_conf:TestConfig
    
    def test_norm_weights_pass(self):
        """Vanilla IS with non-bias averaging:
        $w_{H,i}=\prod_{t=0}^{H}w_{t,i}$
        
        $\frac{1}/{n}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t}\gamma^{t}w_{H,i}$
        
        => The output should be of the form:
        \frac{1}/{n}w_{H,i}
        """
        test_res = self.test_conf.traj_is_weights_is
        toll = test_res.mean()/1000
        calculator = PassWeightDenom()
        pred_res = calculator(
            weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
            )
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())


    def test_norm_weights_pit_is_base(self):
        """Vanilla IS with PIT averaging:
        .. math::
            w_{H,i}=\prod_{t=0}^{H}w_{t,i}
            w_{H} = \frac{1}{n_{t}}\sum_{i=1}^{n_{t}} \mathbbm{1}w_{i,t}>0w_{H,i}
            n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{H,i}\frac{1}/{w_{H}}
        The output should be of the form:
        .. math::
            \sum_{i=1}^{n_{t}} \mathbbm{1}w_{i,t}>0w_{H,i}
        """
        # test_conf.traj_is_weights_is defines the Vanilla IS one step weights 
        # i.e., w_{H,i}
        # Summing to define \sum_{i=1}^{n}\prod_{t=0}^{H}w_{t,i}
        # The input weights are the same for all steps in a trajectory, 
        # therefore, sum across the trajectories
        
        # Find the final weight for each trajectory
        max_h = max([len(i) for i in self.test_conf.test_act_indiv_weights])
        weights = [
                torch.tensor(i) for i in self.test_conf.test_act_indiv_weights
                ]
        test_res_w = get_wpd_denoms(
            weights=weights,
            h=max_h,
            is_type="is",
            agg_type="sum"
            )
        toll = test_res_w.mean()/1000
        weight_pad = [
            pad(
                w.prod(0,keepdim=True).repeat(w.shape[0]), 
                (0,max_h-w.shape[0])
                )[None,:]
            for w in weights]
        weight_pad = torch.concat(weight_pad, dim=0)
        calculator = SumPiTWeightDenom()
        pred_res1 = calculator.get_pit_denom(weight_pad)
        calculator = AvgPiTWeightDenom()
        pred_res2 = calculator.get_pit_denom(weight_pad)
        np.testing.assert_allclose(pred_res1.numpy(), test_res_w.numpy(), 
                                atol=toll.numpy())
        np.testing.assert_allclose(pred_res2.numpy(), test_res_w.numpy(), 
                                atol=toll.numpy())

    def test_norm_weights_pit_is_smooth_base(self):
        """Vanilla IS with PIT averaging:
        .. math::
            w_{H,i}=\prod_{t=0}^{H}w_{t,i}
            w_{H} = \frac{1}{n_{t}}\sum_{i=1}^{n_{t}} \mathbbm{1}w_{i,t}>0w_{H,i}
            n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{H,i}\frac{1}/{w_{H}}
        The output should be of the form:
        .. math::
            \sum_{i=1}^{n_{t}} \mathbbm{1}w_{i,t}>0w_{H,i}
        """
        # test_conf.traj_is_weights_is defines the Vanilla IS one step weights 
        # i.e., w_{H,i}
        # Summing to define \sum_{i=1}^{n}\prod_{t=0}^{H}w_{t,i}
        # The input weights are the same for all steps in a trajectory, 
        # therefore, sum across the trajectories
        
        # Find the final weight for each trajectory
        smooth_eps = 0.00001
        max_h = max([len(i) for i in self.test_conf.test_act_indiv_weights])
        weights = [
                torch.tensor(i) for i in self.test_conf.test_act_indiv_weights
                ]
        test_res_w = get_wpd_denoms(
            weights=weights,
            h=max_h,
            is_type="is",
            agg_type="sum"
            )
        test_res_w = test_res_w+smooth_eps
        toll = test_res_w.mean()/1000
        weight_pad = [
            pad(
                w.prod(0,keepdim=True).repeat(w.shape[0]), 
                (0,max_h-w.shape[0])
                )[None,:]
            for w in weights]
        weight_pad = torch.concat(weight_pad, dim=0)
        calculator = SumPiTWeightDenom(smooth_eps=smooth_eps)
        pred_res1 = calculator.get_pit_denom(weight_pad)
        calculator = AvgPiTWeightDenom(smooth_eps=smooth_eps)
        pred_res2 = calculator.get_pit_denom(weight_pad)
        np.testing.assert_allclose(pred_res1.numpy(), test_res_w.numpy(), 
                                atol=toll.numpy())
        np.testing.assert_allclose(pred_res2.numpy(), test_res_w.numpy(), 
                                atol=toll.numpy())
        

    def test_norm_weights_pit_is_sum(self):
        """Vanilla IS with PIT averaging:
        .. math::
            w_{H,i}=\prod_{t=0}^{H}w_{t,i}
            w_{H} = \frac{1}{n_{t}}\sum_{i=1}^{n_{t}} \mathbbm{1}w_{i,t}>0w_{H,i}
            n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{H,i}\frac{1}/{w_{H}}
        The output should be of the form:
        .. math::
            w_{H,i}\frac{1}/{\sum_{i=1}^{n_{t}} \mathbbm{1}w_{i,t}>0w_{H,i}}
        """
        # test_conf.traj_is_weights_is defines the Vanilla IS one step weights 
        # i.e., w_{H,i}
        # Summing to define \sum_{i=1}^{n}\prod_{t=0}^{H}w_{t,i}
        # The input weights are the same for all steps in a trajectory, 
        # therefore, sum across the trajectories
        
        # Find the final weight for each trajectory
        max_h = max([len(i) for i in self.test_conf.test_act_indiv_weights])
        weights = [
                torch.tensor(i) for i in self.test_conf.test_act_indiv_weights
                ]
        test_res_w = get_wpd_denoms(
            weights=weights,
            h=max_h,
            is_type="is",
            agg_type="sum"
            )
        weight_pad = [
            pad(
                w.prod(0,keepdim=True).repeat(w.shape[0]), 
                (0,max_h-w.shape[0])
                )[None,:]
            for w in weights]
        weight_pad = torch.concat(weight_pad, dim=0)
        test_res = weight_pad/test_res_w
        toll = test_res.mean()/1000
        calculator = SumPiTWeightDenom()
        pred_res = calculator(
            weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())


    def test_norm_weights_pit_pd_sum(self):
        """Vanilla IS with PIS averaging:
        .. math::
            w_{0:t',i}=\prod_{t=0}^{t'}w_{t,i}
            w_{0:t'} = \sum_{i=1}^{n_{t'}} \mathbbm{1}[w_{i,t}>0]w_{0:t',i}
            n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{0:t,i}\frac{1}/{n_{t}^{-1}w_{0:t}}
        The output should be of the form:
        .. math::
            w_{0:t,i}\frac{1}/{w_{0:t}}
        """
        # test_conf.traj_is_weights_is defines the Vanilla IS one step weights 
        # i.e., w_{H,i}
        # Summing to define \sum_{i=1}^{n}\prod_{t=0}^{H}w_{t,i}
        # The input weights are the same for all steps in a trajectory, 
        # therefore, sum across the trajectories
        
        # Find the final weight for each trajectory
        max_h = max([len(i) for i in self.test_conf.test_act_indiv_weights])
        weights = [
                torch.tensor(i) for i in self.test_conf.test_act_indiv_weights
                ]
        test_res_w = get_wpd_denoms(
            weights=weights,
            h=max_h,
            is_type="pd",
            agg_type="sum"
            )
        weight_pad = [
            pad(
                w.cumprod(0), 
                (0,max_h-w.shape[0])
                )[None,:]
            for w in weights]
        weight_pad = torch.concat(weight_pad, dim=0)
        test_res = weight_pad/test_res_w
        toll = test_res.mean()/1000
        calculator = SumPiTWeightDenom()
        pred_res = calculator(
            weights=self.test_conf.traj_is_weights_pd, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())


    def test_norm_weights_pit_is_avg(self):
        """Vanilla IS with PIT averaging:
        .. math::
            w_{H,i}=\prod_{t=0}^{H}w_{t,i}
            w_{H} = \frac{1}{n_{t}}\sum_{i=1}^{n_{t}} \mathbbm{1}w_{i,t}>0w_{H,i}
            n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{H,i}\frac{1}/{w_{H}}
        The output should be of the form:
        .. math::
            w_{H,i}\frac{1}/{\sum_{i=1}^{n_{t}} \mathbbm{1}w_{i,t}>0w_{H,i}}
        """
        # test_conf.traj_is_weights_is defines the Vanilla IS one step weights 
        # i.e., w_{H,i}
        # Summing to define \sum_{i=1}^{n}\prod_{t=0}^{H}w_{t,i}
        # The input weights are the same for all steps in a trajectory, 
        # therefore, sum across the trajectories
        
        # Find the final weight for each trajectory
        max_h = max([len(i) for i in self.test_conf.test_act_indiv_weights])
        weights = [
                torch.tensor(i) for i in self.test_conf.test_act_indiv_weights
                ]
        test_res_w = get_wpd_denoms(
            weights=weights,
            h=max_h,
            is_type="is",
            agg_type="mean"
            )
        weight_pad = [
            pad(
                w.prod(0,keepdim=True).repeat(w.shape[0]), 
                (0,max_h-w.shape[0])
                )[None,:]
            for w in weights]
        weight_pad = torch.concat(weight_pad, dim=0)
        test_res = weight_pad/test_res_w
        toll = test_res.mean()/1000
        calculator = AvgPiTWeightDenom()
        pred_res = calculator(
            weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())


    def test_norm_weights_pit_pd_avg(self):
        """Vanilla IS with PIS averaging:
        .. math::
            w_{0:t',i}=\prod_{t=0}^{t'}w_{t,i}
            w_{0:t'} = \sum_{i=1}^{n_{t'}} \mathbbm{1}[w_{i,t}>0]w_{0:t',i}
            n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{0:t,i}\frac{1}/{n_{t}^{-1}w_{0:t}}
        The output should be of the form:
        .. math::
            w_{0:t,i}\frac{1}/{w_{0:t}}
        """
        # test_conf.traj_is_weights_is defines the Vanilla IS one step weights 
        # i.e., w_{H,i}
        # Summing to define \sum_{i=1}^{n}\prod_{t=0}^{H}w_{t,i}
        # The input weights are the same for all steps in a trajectory, 
        # therefore, sum across the trajectories
        
        # Find the final weight for each trajectory
        max_h = max([len(i) for i in self.test_conf.test_act_indiv_weights])
        weights = [
                torch.tensor(i) for i in self.test_conf.test_act_indiv_weights
                ]
        test_res_w = get_wpd_denoms(
            weights=weights,
            h=max_h,
            is_type="pd",
            agg_type="mean"
            )
        weight_pad = [
            pad(
                w.cumprod(0), 
                (0,max_h-w.shape[0])
                )[None,:]
            for w in weights]
        weight_pad = torch.concat(weight_pad, dim=0)
        test_res = weight_pad/test_res_w
        toll = test_res.mean()/1000
        calculator = AvgPiTWeightDenom()
        pred_res = calculator(
            weights=self.test_conf.traj_is_weights_pd, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())


    def test_norm_weights_avg_is(self):
        """Vanilla IS:
        .. math::
            w_{H,i}=\prod_{t=0}^{H}w_{t,i}
            n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{H,i}
        The output should be of the form:
        .. math::
            \frac{1}/{n}w_{H,i}
        """
        # test_conf.traj_is_weights_is defines the Vanilla IS one step weights 
        # i.e., w_{H,i}
        # Summing to define \sum_{i=1}^{n}\prod_{t=0}^{H}w_{t,i}
        # The input weights are the same for all steps in a trajectory, 
        # therefore, sum across the trajectories
        
        # Find the final weight for each trajectory
        max_h = max([len(i) for i in self.test_conf.test_act_indiv_weights])
        weights = [
                torch.tensor(i) for i in self.test_conf.test_act_indiv_weights
                ]
        weight_pad = [
            pad(
                w.prod(0,keepdim=True).repeat(w.shape[0]), 
                (0,max_h-w.shape[0])
                )[None,:]
            for w in weights]
        weight_pad = torch.concat(weight_pad, dim=0)
        test_res = weight_pad/len(weights)
        toll = test_res.mean()/1000
        calculator = AvgWeightDenom()
        pred_res = calculator(
            weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())

    def test_norm_weights_avg_pd(self):
        """Vanilla IS with PIS averaging:
        .. math::
            w_{0:t',i}=\prod_{t=0}^{t'}w_{t,i}
            n^{-1}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{0:t,i}
        The output should be of the form:
        .. math::
             \frac{1}/{n}w_{0:t',i}
        """
        # test_conf.traj_is_weights_is defines the Vanilla IS one step weights 
        # i.e., w_{H,i}
        # Summing to define \sum_{i=1}^{n}\prod_{t=0}^{H}w_{t,i}
        # The input weights are the same for all steps in a trajectory, 
        # therefore, sum across the trajectories
        
        # Find the final weight for each trajectory
        max_h = max([len(i) for i in self.test_conf.test_act_indiv_weights])
        weights = [
                torch.tensor(i) for i in self.test_conf.test_act_indiv_weights
                ]
        weight_pad = [
            pad(
                w.cumprod(0), 
                (0,max_h-w.shape[0])
                )[None,:]
            for w in weights]
        weight_pad = torch.concat(weight_pad, dim=0)
        test_res = weight_pad/len(weights)
        toll = test_res.mean()/1000
        calculator = AvgWeightDenom()
        pred_res = calculator(
            weights=self.test_conf.traj_is_weights_pd, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())

