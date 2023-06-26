import logging
import numpy as np
import torch
from typing import Dict, List, Callable
from d3rlpy.metrics.scorer import AlgoProtocol
from d3rlpy.dataset import Episode

from ...is_pipeline import eval_weight_array
from ...components.Policy import Policy, D3RlPyDeterministic
from ...components.ImportanceSampler import ISWeightOrchestrator
from ...OPEEstimators import ISEstimator
from .misc_scorers import QueryScorer
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
        
        # weight_res = []
        # discnt_reward_res = []
        # norm_conts = []
        # # For each trajectory in an input dataset, get a list of the weights
        # # and rewards for each trajectory
        # # TODO: Consolodate with is_pipeline code!
        # for episode in self.episodes:
        #     weight, discnt_reward, norm_cont = is_calculator.get_traj_w_r(
        #         state=torch.Tensor(episode.observations), 
        #         action=torch.Tensor(episode.actions.reshape(-1,1)), 
        #         reward=torch.Tensor(episode.rewards.reshape(-1,1))
        #         )
        #     weight_res.append(weight)
        #     discnt_reward_res.append(discnt_reward)
        #     norm_conts.append(norm_cont)

        # if self.norm_weights:
        #     norm_val = torch.sum(torch.Tensor(norm_conts))
        #     weight_res = [w/norm_val for w in weight_res]
        # else:
        #     weight_res = [w/len(self.episodes) for w in weight_res]
        
        # weight_res = torch.concat(weight_res)
        # discnt_reward_res = torch.concat(discnt_reward_res)
    
        # loss, _, clip_loss, _, weight_res, clip_weight_res = eval_weight_array(
        #     weight_res=weight_res, discnt_reward_res=discnt_reward_res,
        #     save_dir=None, prefix=None, clip=self.clip)
        # # TODO: Assumes 1 dimensional action!
        # eval_policy_acts = [tens.squeeze().detach().numpy().reshape(-1) 
        #                     for tens in eval_policy.policy_actions]
        # eval_policy_acts = np.concatenate(eval_policy_acts)
        # action_dist = {}
        # for i in self.unique_pol_acts:
        #     action_dist[i] = (
        #         (eval_policy_acts == i).sum()/len(eval_policy_acts)
        #     )
        # res = {
        #     "action_dist": action_dist,
        #     "loss": loss,
        #     "weight_res_mean": weight_res.mean(), 
        #     "weight_res_std": weight_res.std()
        # }
        # return res

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
        