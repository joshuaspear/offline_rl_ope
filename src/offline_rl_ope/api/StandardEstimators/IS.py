from ...OPEEstimators.IS import ISEstimator
from ...OPEEstimators.WeightDenom import (
    PassWeightDenom, AvgPiTWeightDenom
    )
from ...OPEEstimators.EmpiricalMeanDenom import (
    EmpiricalMeanDenom, 
    )

class VanillaISPDIS(ISEstimator):
    
    def __init__(
        self, 
        clip_weights: bool = False, 
        clip: float = 0.0, 
        cache_traj_rewards: bool = False
        ) -> None:
        """_summary_
        - http://incompleteideas.net/papers/PSS-00.pdf (Q^{IS} when weights are IS)
        - http://incompleteideas.net/papers/PSS-00.pdf (Q^{PD} when weights are PD)
        - https://arxiv.org/pdf/1906.03735 (snis when weights are IS)
        - https://arxiv.org/pdf/1906.03735 (snsis when weights are PD)
        Args:
            clip_weights (bool, optional): _description_. Defaults to False.
            clip (float, optional): _description_. Defaults to 0.
            cache_traj_rewards (bool, optional): _description_. Defaults to False.
        """
        super().__init__(
            empirical_denom=EmpiricalMeanDenom(),
            weight_denom=PassWeightDenom(), 
            clip_weights=clip_weights, 
            clip=clip, 
            cache_traj_rewards=cache_traj_rewards
            )

class WeightedISPDIS(ISEstimator):
    
    def __init__(
        self, 
        clip_weights: bool = False, 
        clip: float = 0.0, 
        cache_traj_rewards: bool = False,
        smooth_eps:float = 0.0
        ):
        """
        - http://incompleteideas.net/papers/PSS-00.pdf (Q^{ISW} when weights are IS)
        - https://arxiv.org/pdf/1906.03735 (snsis when weights are PD)
        Args:
            clip_weights (bool, optional): _description_. Defaults to False.
            clip (float, optional): _description_. Defaults to 0.
            cache_traj_rewards (bool, optional): _description_. Defaults to False.
            smooth_eps (float, optional): _description_. Defaults to 0.0.
        """
        empirical_denom = EmpiricalMeanDenom()
        weight_denom = AvgPiTWeightDenom(smooth_eps=smooth_eps)
        super().__init__(
            empirical_denom, 
            weight_denom, 
            clip_weights, 
            clip, 
            cache_traj_rewards
            )