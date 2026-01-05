import numpy as np
import torch
from torch.nn.functional import pad
from typing import Any, List, Dict
import math
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker

from ...OPEEstimators import (
    DirectMethodBase, 
    DREstimator,
    EmpiricalMeanDenom,
    PassWeightDenom,
    AvgPiTWeightDenom,
    )

class VanillaDR(DREstimator):
    
    def __init__(
        self, 
        dm_model: DirectMethodBase, 
        clip_weights: bool = False, 
        clip: float = 0.0, 
        cache_traj_rewards: bool = False, 
        ) -> None:
        """_summary_
        - https://arxiv.org/pdf/1906.03735 (\beta_{dr} when weights are IS)
        Args:
            dm_model (DirectMethodBase): _description_
            clip_weights (bool, optional): _description_. Defaults to False.
            clip (float, optional): _description_. Defaults to 0.0.
            cache_traj_rewards (bool, optional): _description_. Defaults to False.
        """
        super().__init__(
            empirical_denom=EmpiricalMeanDenom(),
            weight_denom=PassWeightDenom(),
            dm_model=dm_model, 
            clip_weights=clip_weights, 
            clip=clip, 
            cache_traj_rewards=cache_traj_rewards, 
            )
        
class WDR(DREstimator):
    
    def __init__(
        self, 
        dm_model: DirectMethodBase,  
        clip_weights: bool = False, 
        clip: float = 0.0, 
        cache_traj_rewards: bool = False, 
        smooth_eps:float=0.0
        ) -> None:
        """_summary_
        - https://arxiv.org/pdf/1906.03735 (\beta_{wdr} when weights are PD)
        Args:
            dm_model (DirectMethodBase): _description_
            clip_weights (bool, optional): _description_. Defaults to False.
            clip (float, optional): _description_. Defaults to 0.0.
            cache_traj_rewards (bool, optional): _description_. Defaults to False.
            smooth_eps (float, optional): _description_. Defaults to 0.0.
        """
        super().__init__(
            empirical_denom=EmpiricalMeanDenom(),
            weight_denom=AvgPiTWeightDenom(smooth_eps=smooth_eps),
            dm_model=dm_model, 
            clip_weights=clip_weights, 
            clip=clip, 
            cache_traj_rewards=cache_traj_rewards, 
            )