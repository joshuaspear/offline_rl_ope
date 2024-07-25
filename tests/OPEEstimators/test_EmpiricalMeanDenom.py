import copy
import numpy as np
import torch
import unittest
from offline_rl_ope.OPEEstimators.EmpiricalMeanDenom import (
    EmpiricalMeanDenom, WeightedEmpiricalMeanDenom
)
from parameterized import parameterized_class
from ..base import test_configs_fmt_class, TestConfig

@parameterized_class(test_configs_fmt_class)
class UtilsTestEmpiricalMeanDenom(unittest.TestCase):
    
    test_conf:TestConfig
    
    def test_norm_weights_vanilla(self):
        """Vanilla IS with non-bias averaging:
        $w_{H,i}=\prod_{t=0}^{H}w_{t,i}$
        
        $\frac{1}/{n}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t}\gamma^{t}w_{H,i}$
        
        => The output should be of the form:
        n
        """
        test_res = self.test_conf.traj_is_weights_is.shape[0]
        toll = test_res/1000
        calculator = EmpiricalMeanDenom()
        assert len(self.test_conf.traj_is_weights_is.shape) == 2, "Incorrect test input dimensions"
        assert len(self.test_conf.msk_test_res.shape) == 2, "Incorrect test input dimensions"
        pred_res = calculator(
            weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
            )
        np.testing.assert_allclose(
            pred_res.numpy(), np.array(test_res), atol=np.array(toll)
            )
        
    def test_norm_weights_wis(self):
        """Vanilla IS with WIS averaging:
        .. math::
            w_{H,i}=\prod_{t=0}^{H}w_{t,i}
            w_{H} = \sum_{i=1}^{n} w_{H,i}
            $\frac{1}/{w_{H}}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{H,i}$
        The output should be of the form:
        .. math::
            w_{H}
        """
        
        # Find the final weight for each trajectory
        term_idx = [len(i) for i in self.test_conf.test_act_indiv_weights]
        term_weights = []
        for idx, traj in zip(term_idx, self.test_conf.traj_is_weights_is):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum()
        denom_toll = denom.squeeze().numpy()/1000
        calculator = WeightedEmpiricalMeanDenom()
        norm = calculator(
            weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            np.array(norm), denom.numpy(), 
            atol=denom_toll
            )
        
    def test_norm_weights_wis_smooth(self):
        """Vanilla IS with WIS averaging:
        .. math::
            w_{H,i}=\prod_{t=0}^{H}w_{t,i}
            w_{H} = \sum_{i=1}^{n} w_{H,i}
            $\frac{1}/{\eps + w_{H}}\sum_{i=1}^{n}\sum_{t=0}^{H}r_{t,i}\gamma^{t}w_{H,i}$
        The output should be of the form:
        .. math::
            \eps + w_{H}
        """
        
        # Find the final weight for each trajectory
        smooth_eps = 0.00001
        term_idx = [len(i) for i in self.test_conf.test_act_indiv_weights]
        term_weights = []
        for idx, traj in zip(term_idx, self.test_conf.traj_is_weights_is):
            term_weights.append(traj[idx-1])
        term_weights = torch.tensor(term_weights)
        # Sum over the weights as we are not doing cumulative
        denom = term_weights.sum() + smooth_eps
        denom_toll = denom.squeeze().numpy()/1000
        calculator = WeightedEmpiricalMeanDenom(smooth_eps=smooth_eps)
        norm = calculator(
            weights=self.test_conf.traj_is_weights_is, 
            is_msk=self.test_conf.msk_test_res
        )
        np.testing.assert_allclose(
            np.array(norm), denom.numpy(), 
            atol=denom_toll
            )

