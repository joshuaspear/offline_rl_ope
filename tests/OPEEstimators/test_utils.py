import copy
import numpy as np
import torch
import unittest
from offline_rl_ope.OPEEstimators.utils import (
    clip_weights, clip_weights_pass, VanillaNormWeights, WISWeightNorm)
from parameterized import parameterized_class
from ..base import test_configs_fmt_class, TestConfig

@parameterized_class(test_configs_fmt_class)
class UtilsTestVanillaIS(unittest.TestCase):
    
    test_conf:TestConfig
    
    def setUp(self) -> None:
        self.clip_toll = self.test_conf.weight_test_res.numpy().mean()/1000

    def test_clip_weights(self):
        clip = 1.2
        test_res = self.test_conf.weight_test_res.clamp(max=1.2, min=1/1.2)
        assert len(self.test_conf.weight_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = clip_weights(self.test_conf.weight_test_res, clip=clip)
        self.assertEqual(pred_res.shape,self.test_conf.weight_test_res.shape)
        np.testing.assert_allclose(pred_res, test_res, atol=self.clip_toll)
    
    def test_clip_weights_pass(self):
        clip = 1.2
        test_res = copy.deepcopy(self.test_conf.weight_test_res)
        assert len(self.test_conf.weight_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = clip_weights_pass(self.test_conf.weight_test_res, clip=clip)
        self.assertEqual(pred_res.shape,self.test_conf.weight_test_res.shape)
        np.testing.assert_allclose(pred_res, test_res, atol=self.clip_toll)
        
    # def test_norm_weights_pass(self):
    #     test_res = weight_test_res/msk_test_res.sum(axis=0)
    #     toll = test_res.mean()/1000
    #     pred_res = norm_weights_pass(traj_is_weights=weight_test_res, 
    #                                  is_msk=msk_test_res)
    #     self.assertEqual(pred_res.shape,weight_test_res.shape)
    #     np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
    #                                atol=toll.numpy())
    
    def test_norm_weights_vanilla(self):
        """Vanilla IS with non-bias averaging:
        $w_{H,i}=\prod_{t=0}^{H}w_{t,i}$
        
        $\frac{1}/{n}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t}\gamma^{t}w_{H,i}$
        
        => The output should be of the form:
        \frac{1}/{n}w_{H,i}
        """
        denom = self.test_conf.traj_is_weights_is.shape[0]
        test_res = self.test_conf.traj_is_weights_is/denom
        toll = test_res.mean()/1000
        calculator = VanillaNormWeights()
        assert len(self.test_conf.traj_is_weights_is.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_is.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())

    def test_norm_weights_vanilla_no_avg(self):
        """Vanilla IS with non-bias averaging:
        $w_{H,i}=\prod_{t=0}^{H}w_{t,i}$
        
        $\frac{1}/{n}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t}\gamma^{t}w_{H,i}$
        
        => The output should be of the form:
        \frac{1}/{n}w_{H,i}
        """
        test_res = self.test_conf.traj_is_weights_is
        toll = test_res.mean()/1000
        calculator = VanillaNormWeights(avg_denom=False)
        assert len(self.test_conf.traj_is_weights_is.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_is.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())

    
    def test_norm_weights_wis(self):
        """Vanilla IS with WIS averaging:
        .. math::
            w_{H,i}=\prod_{t=0}^{H}w_{t,i}
            w_{H} = \sum_{i=1}^{n} w_{H,i}
            $\frac{1}/{w_{H}}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{H,i}$
        The output should be of the form:
        .. math::
            \frac{1}/{w_{H}}w_{H,i}
        """
        # test_conf.traj_is_weights_is defines the Vanilla IS one step weights 
        # i.e., w_{H,i}
        # Summing to define \sum_{i=1}^{n}\prod_{t=0}^{H}w_{t,i}
        # The input weights are the same for all steps in a trajectory, 
        # therefore, sum across the trajectories
        
        # Find the final weight for each trajectory
        term_idx = [len(i) for i in self.test_conf.test_act_indiv_weights]
        term_weights = []
        for idx, traj in zip(term_idx, self.test_conf.traj_is_weights_is):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum()
        denom_toll = denom.squeeze().numpy()/1000
        test_res = self.test_conf.traj_is_weights_is/denom
        toll = test_res.mean()/1000
        calculator = WISWeightNorm()
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_is.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_is.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())
        
    def test_norm_weights_wis_cum(self):
        """Vanilla IS with WIS cumulative averaging:
        .. math::
        w_{H,i} = \prod_{t=0}^{H}w_{t,i}
        w_{H,t} = \sum_{i=1}^{n}\sum_{t=0}^{H}w_{H,i}
        \sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}\frac{1}/{w_{H,t}}w_{H,i}
        The output should be of the form:
        .. math::
        $\frac{1}/{w_{H,t}}w_{H,i}$
        """
        # \sum_{i=1}^{n}\sum_{t=0}^{H}w_{H,i}
        denom = self.test_conf.traj_is_weights_is.sum()
        denom_toll = denom.squeeze().numpy()/1000
        test_res = self.test_conf.traj_is_weights_is/denom
        toll = test_res.mean()/1000
        calculator = WISWeightNorm(cumulative=True)
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_is.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_is.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())

    
    def test_norm_weights_wis_smooth(self):
        smooth_eps = 0.00000001
        term_idx = [
            len(i) for i in self.test_conf.test_act_indiv_weights_alter
            ]
        term_weights = []
        for idx, traj in zip(
            term_idx, 
            self.test_conf.traj_is_weights_is_alter
            ):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum() + smooth_eps
        denom_toll = denom.squeeze().numpy()/1000
        test_res = self.test_conf.traj_is_weights_is_alter/denom
        toll = test_res.mean()/1000
        calculator = WISWeightNorm(smooth_eps=smooth_eps)
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_is_alter, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_is_alter.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_is_alter, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_is_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())
        
    
    def test_norm_weights_wis_no_smooth(self):
        term_idx = [
            len(i) for i in self.test_conf.test_act_indiv_weights_alter
            ]
        term_weights = []
        for idx, traj in zip(
            term_idx, 
            self.test_conf.traj_is_weights_is_alter
            ):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum()
        denom_toll = denom.squeeze().numpy()/1000
        test_res = self.test_conf.traj_is_weights_is_alter/denom
        toll = test_res.mean()/1000
        calculator = WISWeightNorm()
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_is_alter, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_is_alter.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_is_alter, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_is_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())

    def test_norm_weights_wis_smooth_avg(self):
        smooth_eps = 0.00000001
        term_idx = [
            len(i) for i in self.test_conf.test_act_indiv_weights_alter
            ]
        term_weights = []
        for idx, traj in zip(
            term_idx, 
            self.test_conf.traj_is_weights_is_alter
            ):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum() + smooth_eps
        denom_toll = denom.squeeze().mean().numpy()/1000
        test_res = (self.test_conf.traj_is_weights_is_alter/denom)/len(term_idx)
        toll = test_res.mean()/1000
        calculator = WISWeightNorm(smooth_eps=smooth_eps, avg_denom=True)
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_is_alter, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_is_alter.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_is_alter, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_is_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())

    def test_norm_weights_wis_no_smooth_avg(self):
        term_idx = [
            len(i) for i in self.test_conf.test_act_indiv_weights_alter
            ]
        term_weights = []
        for idx, traj in zip(
            term_idx, 
            self.test_conf.traj_is_weights_is_alter
            ):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum()
        denom_toll = denom.squeeze().mean().numpy()/1000
        test_res = (self.test_conf.traj_is_weights_is_alter/denom)/len(term_idx)
        toll = test_res.mean()/1000
        calculator = WISWeightNorm(avg_denom=True)
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_is_alter, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_is_alter.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_is_alter, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_is_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())
        
@parameterized_class(test_configs_fmt_class)
class UtilsTestPD(unittest.TestCase):
    
    test_conf:TestConfig
    
    def test_norm_weights_vanilla(self):
        """PD with non-bias averaging:
        $\frac{1}/{n}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t}\gamma^{t}\prod_{t=0}^{t'}w_{t,i}$
        
        => The output should be of the form:
        $\frac{1}/{n}\prod_{t=0}^{t'}w_{t,i}$
        """
        denom = self.test_conf.traj_is_weights_pd.shape[0]
        test_res = self.test_conf.traj_is_weights_pd/denom
        toll = test_res.mean()/1000
        calculator = VanillaNormWeights()
        assert len(self.test_conf.traj_is_weights_pd.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_pd, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_pd.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())
        
    def test_norm_weights_vanilla_no_avg(self):
        """PD with non-bias averaging:
        $\frac{1}/{n}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t}\gamma^{t}\prod_{t=0}^{t'}w_{t,i}$
        
        => The output should be of the form:
        $\frac{1}/{n}\prod_{t=0}^{t'}w_{t,i}$
        """
        test_res = self.test_conf.traj_is_weights_pd
        toll = test_res.mean()/1000
        calculator = VanillaNormWeights(avg_denom=False)
        assert len(self.test_conf.traj_is_weights_pd.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_pd, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_pd.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())

    
    def test_norm_weights_wpd(self):
        """WPD:
        w_{H,i}=\prod_{t=0}^{H}w_{t,i}
        w_{H} = \sum_{i=1}^{n} w_{H,i}
        \frac{1}/{w_{H}}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}\prod_{t=0}^{t'}w_{t,i}
        The output should be of the form:
        \frac{1}/{w_{H}}\prod_{t=0}^{t'}w_{t,i}
        """
        term_idx = [len(i) for i in self.test_conf.test_act_indiv_weights]
        term_weights = []
        for idx, traj in zip(term_idx, self.test_conf.traj_is_weights_pd):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum()
        denom_toll = denom.squeeze().numpy()/1000
        test_res = self.test_conf.traj_is_weights_pd/denom
        toll = test_res.mean()/1000
        calculator = WISWeightNorm()
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_pd, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_pd.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_pd, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_pd.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())
        
    def test_norm_weights_wpd_cum(self):
        """WPD:
        w_{t',i}=\prod_{t=0}^{t'}w_{t,i}
        w_{t'} = \sum_{i=1}^{n} w_{t',i}
        $\frac{1}/{w_{t'}}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}\prod_{t=0}^{t'}w_{t,i}$
        => The output should be of the form:
        \frac{1}/{w_{t'}}\prod_{t=0}^{t'}w_{t,i}
        """
        # Sum across the trajectories to get the time t cumulative weight
        # Note, the weight is already cumulative due to PD input
        denom = self.test_conf.traj_is_weights_pd.sum()
        # No need to alter shape
        denom_toll = denom.squeeze().numpy()/1000
        test_res = self.test_conf.traj_is_weights_pd/denom
        toll = test_res.mean()/1000
        calculator = WISWeightNorm(cumulative=True)
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_pd, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_pd.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_pd, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_pd.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())

    
    def test_norm_weights_wpd_smooth(self):
        smooth_eps = 0.00000001
        term_idx = [
            len(i) for i in self.test_conf.test_act_indiv_weights_alter
            ]
        term_weights = []
        for idx, traj in zip(
            term_idx, 
            self.test_conf.traj_is_weights_pd_alter
            ):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum() + smooth_eps
        denom_toll = denom.squeeze().numpy()/1000
        test_res = self.test_conf.traj_is_weights_pd_alter/denom
        toll = test_res.mean()/1000
        calculator = WISWeightNorm(smooth_eps=smooth_eps)
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_pd_alter, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_pd_alter.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_pd_alter, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_pd_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())
        
    
    def test_norm_weights_wpd_no_smooth(self):
        term_idx = [
            len(i) for i in self.test_conf.test_act_indiv_weights_alter
            ]
        term_weights = []
        for idx, traj in zip(
            term_idx, 
            self.test_conf.traj_is_weights_pd_alter
            ):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum()
        denom_toll = denom.squeeze().numpy()/1000
        test_res = self.test_conf.traj_is_weights_pd_alter/denom
        toll = test_res.mean()/1000
        calculator = WISWeightNorm()
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_pd_alter, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_pd_alter.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_pd_alter, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_pd_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())

    def test_norm_weights_wpd_smooth_avg(self):
        smooth_eps = 0.00000001
        term_idx = [
            len(i) for i in self.test_conf.test_act_indiv_weights_alter
            ]
        term_weights = []
        for idx, traj in zip(
            term_idx, 
            self.test_conf.traj_is_weights_pd_alter
            ):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum() + smooth_eps
        denom_toll = denom.squeeze().numpy()/1000
        test_res = (self.test_conf.traj_is_weights_pd_alter/denom)/len(term_idx)
        toll = test_res.mean()/1000
        calculator = WISWeightNorm(smooth_eps=smooth_eps, avg_denom=True)
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_pd_alter, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_pd_alter.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_pd_alter, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_pd_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())

    def test_norm_weights_wpd_no_smooth_avg(self):
        term_idx = [
            len(i) for i in self.test_conf.test_act_indiv_weights_alter
            ]
        term_weights = []
        for idx, traj in zip(
            term_idx, 
            self.test_conf.traj_is_weights_pd_alter
            ):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum()
        denom_toll = denom.squeeze().numpy()/1000
        test_res = (self.test_conf.traj_is_weights_pd_alter/denom)/len(term_idx)
        toll = test_res.mean()/1000
        calculator = WISWeightNorm(avg_denom=True)
        norm = calculator.calc_norm(
            traj_is_weights=self.test_conf.traj_is_weights_pd_alter, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            norm.numpy(), denom.numpy(), 
            atol=denom_toll
            )
        assert len(self.test_conf.traj_is_weights_pd_alter.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            traj_is_weights=self.test_conf.traj_is_weights_pd_alter, 
            is_msk=self.test_conf.msk_test_res
            )
        self.assertEqual(pred_res.shape,self.test_conf.traj_is_weights_pd_alter.shape)
        np.testing.assert_allclose(pred_res.numpy(), test_res.numpy(), 
                                atol=toll.numpy())