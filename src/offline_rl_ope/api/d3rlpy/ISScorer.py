import logging
import torch
from typing import Dict, List, Callable
from d3rlpy.metrics.scorer import AlgoProtocol
from d3rlpy.dataset import Episode

from ...components.Policy import Policy, D3RlPyDeterministic
from ...components.ImportanceSampler import ISWeightOrchestrator
from ...OPEEstimators import ISEstimator
from .utils import OPECallbackBase, OPEEstimatorScorerBase

logger = logging.getLogger("offline_rl_ope")

class D3RlPyTorchAlgoPredict:
    
    def __init__(self, predict_func:Callable):
        self.predict_func = predict_func
        
    def __call__(self, x):
        return torch.Tensor(self.predict_func(x))
    

class ISCallback(ISWeightOrchestrator, OPECallbackBase):
    """Wrapper class for performing importance sampling
    """
    def __init__(self, is_types:List[str], behav_policy: Policy, 
                 episodes: List[Episode], gpu:bool=False, collect_act:bool=False
                 ) -> None:
        OPECallbackBase.__init__(self)
        ISWeightOrchestrator.__init__(self, *is_types, 
                                      behav_policy=behav_policy)
        self.states:List[torch.Tensor] = []
        self.actions:List[torch.Tensor] = []
        self.rewards:List[torch.Tensor] = []
        for traj in episodes:
            self.states.append(torch.Tensor(traj.observations))
            self.actions.append(torch.Tensor(traj.actions).view(-1,1))
            self.rewards.append(torch.Tensor(traj.rewards))
        self.gpu = gpu
        self.collect_act = collect_act
        
    def __call__(self, algo: AlgoProtocol, epoch, total_step) -> Dict:
        policy_class = D3RlPyTorchAlgoPredict(
            predict_func=algo.predict)
        eval_policy = D3RlPyDeterministic(policy_class=policy_class, 
                                          collect_res=False, 
                                          collect_act=self.collect_act,
                                          gpu=self.gpu)
        self.update(states=self.states, actions=self.actions, 
                    eval_policy=eval_policy)
        
class ISEstimatorScorer(OPEEstimatorScorerBase, ISEstimator):
    
    def __init__(self, discount, cache:ISCallback, is_type:str, 
                 norm_weights: bool, clip: float = None
                 ) -> None:
        OPEEstimatorScorerBase.__init__(self, cache=cache)
        ISEstimator.__init__(self, norm_weights=norm_weights, clip=clip)
        self.is_type = is_type
        self.discount = discount
        
    def __call__(self, algo: AlgoProtocol, episodes: List[Episode]):
        rewards = [torch.Tensor(ep.rewards) for ep in episodes]
        states = [torch.Tensor(ep.observations) for ep in episodes]
        actions = [torch.Tensor(ep.actions).view(-1,1) for ep in episodes]
        res = self.predict(rewards=rewards, states=states, actions=actions,
                           weights=self.cache[self.is_type].traj_is_weights, 
                           is_msk=self.cache.weight_msk, discount=self.discount
                           )
        return res



class ISDiscreteActionDistScorer:
    
    def __init__(self, cache: ISCallback, act: int) -> None:
        self.cache = cache
        self.act = act
        
    def __call__(self, algo: AlgoProtocol, episodes: List[Episode]):
        all_acts = torch.concat(self.cache.policy_actions).squeeze()
        if len(all_acts) == 0:
            logger.warning(
                "Ensure IS Callback object has been set to track policy actions"
                )
        return (all_acts == self.act).sum()/len(all_acts)
        