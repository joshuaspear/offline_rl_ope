import torch
from typing import Tuple, Literal, Union
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker
from scipy.stats import bootstrap
from scipy.stats._resampling import BootstrapResult
import numpy as np

from .base import Interval
from ..OPEEstimators.IS import ISEstimatorBase
from ..OPEEstimators.EmpiricalMeanDenom import EmpiricalMeanDenom
from ..OPEEstimators.WeightDenom import PassWeightDenom

__all__ = ["BoostrapMean"]

class BoostrapMean(Interval):
    
    def __init__(
        self, 
        n_resamples:int = 9999,
        method:Union[Literal["percentile"],Literal["basic"],Literal["BCa"]] = "BCa"
        ):
        self.__n_resamples = n_resamples
        self.__method = method
    
    @jaxtyped(typechecker=typechecker)
    def predict(
        self,
        is_estimator:ISEstimatorBase,
        confidence:float=0.95
        )->Tuple[float,float]:
        try:
            assert is_estimator.traj_rewards_cache.numel() != 0
        except AssertionError as e:
            e(
                f"""
                Rewards need to be cached! Set 'cache_traj_rewards' to True
                """
            )
        traj_rew = is_estimator.traj_rewards_cache.numpy()
        assert isinstance(traj_rew, np.ndarray)
        res = bootstrap(
            (traj_rew,), 
            np.mean, 
            confidence_level=confidence, 
            rng=np.random.default_rng(),
            alternative="two-sided",
            n_resamples=self.__n_resamples,
            method=self.__method
            )
        assert isinstance(res, BootstrapResult)
        return (
            res.confidence_interval.low,
            res.confidence_interval.high
        )