import torch
from typing import Tuple
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker
from scipy.stats import bootstrap
import numpy as np

from .base import Interval
from ..OPEEstimators.IS import ISEstimatorBase
from ..OPEEstimators.EmpiricalMeanDenom import EmpiricalMeanDenom
from ..OPEEstimators.WeightDenom import PassWeightDenom


class BoostrapMean(Interval):
    
    def __init__(self):
        pass
    
    @jaxtyped(typechecker=typechecker)
    def predict(
        self,
        is_estimator:ISEstimatorBase,
        confidence:float=0.95
        )->Tuple[float,float]:
        try:
            assert is_estimator.traj_rewards_cache != torch.Tensor(0)
        except AssertionError as e:
            e(
                f"""
                Rewards need to be cached! Set 'cache_traj_rewards' to True
                """
            )
        traj_rew = is_estimator.traj_rewards_cache.values()
        res = bootstrap(
            traj_rew, 
            np.mean, 
            confidence_level=confidence, 
            rng=np.random.default_rng()
            )
        try:
            assert hasattr(res, "low")
            assert hasattr(res, "high")
        except AssertionError as e:
            raise TypeError(
                "boostrap result should have attributes 'low' and 'high' however, one is missing"
            )
        return (
            res.low,
            res.high
        )