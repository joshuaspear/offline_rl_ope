import torch
from typing import Tuple
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker
from scipy.stats import t, sem
import numpy as np

from .base import Interval
from ..OPEEstimators.IS import ISEstimatorBase
from ..OPEEstimators.EmpiricalMeanDenom import EmpiricalMeanDenom
from ..OPEEstimators.WeightDenom import PassWeightDenom

from .. import logger

__all__ = ["StudentsT"]

class StudentsT(Interval):
    
    def __init__(self):
        super().__init__()
    
    @jaxtyped(typechecker=typechecker)
    def predict(
        self,
        is_estimator:ISEstimatorBase,
        confidence:float=0.95
        )->Tuple[float,float]:
        emp_denom_valid = isinstance(
            is_estimator.empirical_denom, EmpiricalMeanDenom
            )
        weight_denom_valid = isinstance(
            is_estimator.weight_denom, PassWeightDenom
        )
        if not (emp_denom_valid and weight_denom_valid):
            logger.warning(
                f"""
                Empirical denominator is: {type(is_estimator.empirical_denom)}
                Weight denominator is: {type(is_estimator.weight_denom)}
                Are you sure the confidence interval assumptions are satisfied for Student's T?
                """
                )
        try:
            assert is_estimator.traj_rewards_cache != torch.Tensor(0)
        except AssertionError as e:
            e(
                f"""
                Rewards need to be cached! Set 'cache_traj_rewards' to True
                """
            )
        traj_rew = is_estimator.traj_rewards_cache.values()
        emp_mean = np.mean(traj_rew)
        emp_std = sem(a=traj_rew.values,ddof=1,axis=0)
        n = traj_rew.shape[0]
        dof = n - 1  # degrees of freedom
        confidence_interval = t.interval(
            confidence, dof, emp_mean, emp_std
            )
        return (
            confidence_interval[0],confidence_interval[1]
        )