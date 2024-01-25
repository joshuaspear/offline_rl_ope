import unittest
import torch
import logging
import numpy as np
import copy
from offline_rl_ope.Metrics import EffectiveSampleSize
from ..base import weight_test_res

logger = logging.getLogger("offline_rl_ope")

class TestImportanceSampler:
    
    def __init__(self) -> None:
        self.is_weight_calc = None
        self.traj_is_weights = weight_test_res
        

class EffectiveSampleSizeTest(unittest.TestCase):

    def test_call(self):
        num = torch.sum(torch.pow(weight_test_res,2))
        denum = torch.pow(torch.sum(weight_test_res),2)
        act_res = (num/denum).item()
        metric = EffectiveSampleSize(is_obj=TestImportanceSampler())
        pred_res = metric()
        self.assertEqual(act_res,pred_res)