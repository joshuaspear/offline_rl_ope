import torch
from typing import Tuple, Literal, Union, Callable
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker
from scipy.stats import bootstrap
from scipy.stats._resampling import BootstrapResult
import numpy as np

from .. import logger
from .base import Interval
from ..OPEEstimators.IS import ISEstimatorBase


__all__ = ["BoostrapMean", "BoostrapSum", "Boostrap"]

class Boostrap(Interval):
    
    def __init__(
        self, 
        statistic:Callable,
        n_resamples:int = 9999,
        method:Union[Literal["percentile"],Literal["basic"],Literal["BCa"]] = "BCa"
        ):
        self.__n_resamples = n_resamples
        self.__method = method
        self.__statistic = statistic
        self.__result = None
    
    @property
    def result(self)->BootstrapResult:
        assert self.__result is not None, f"""Run 'predict' method first!"""
        return self.__result
    
    @jaxtyped(typechecker=typechecker)
    def __update(
        self,
        is_estimator:ISEstimatorBase,
        confidence:float=0.95
        )->None:
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
            data=(traj_rew,), 
            # np.mean,
            statistic=self.__statistic,
            confidence_level=confidence, 
            rng=np.random.default_rng(),
            alternative="two-sided",
            n_resamples=self.__n_resamples,
            method=self.__method
            )
        assert isinstance(res, BootstrapResult)
        self.__result = res
        
    
    @jaxtyped(typechecker=typechecker)
    def predict(
        self,
        is_estimator:ISEstimatorBase,
        confidence:float=0.95
        )->Tuple[float,float]:
        self.__update(
            is_estimator=is_estimator, 
            confidence=confidence
        )
        return (
            float(self.result.confidence_interval.low),
            float(self.result.confidence_interval.high)
        )

class BoostrapMean(Boostrap):
    
    def __init__(self, n_resamples = 9999, method = "BCa"):
        super().__init__(
            statistic=np.mean, n_resamples=n_resamples, method=method
            )

    def predict(self, is_estimator, confidence = 0.95):
        return super().predict(is_estimator, confidence)


class BoostrapSum(Boostrap):
    
    def __init__(self, n_resamples = 9999, method = "BCa"):
        super().__init__(
            statistic=np.sum, n_resamples=n_resamples, method=method
            )
    
    def predict(self, is_estimator, confidence = 0.95):
        return super().predict(is_estimator, confidence)