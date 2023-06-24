import numpy as np
import torch
from typing import Dict, List, Callable
from d3rlpy.metrics.scorer import AlgoProtocol
from d3rlpy.dataset import Episode

from ...is_pipeline import eval_weight_array
from ...components.Policy import Policy, D3RlPyDeterministic
from ...components.ImportanceSampler import ImportanceSampler
from .scorers import MultiOutputCache
from .utils import Wrapper

class D3RlPyTorchAlgoPredict:
    
    def __init__(self, predict_func:Callable):
        self.predict_func = predict_func
        
    def __call__(self, x):
        return torch.Tensor(self.predict_func(x))
    

class TorchISEvalD3rlpyWrap(Wrapper):
    """Wrapper class for performing importance sampling
    """
    def __init__(self, importance_sampler:ImportanceSampler,
                 discount:float, behav_policy:Policy, episodes: List[Episode], 
                 norm_weights:bool=False, clip:float=None, is_kwargs={}, 
                 unique_pol_acts = [0,1], gpu:bool=False
                 ) -> None:
        super().__init__(
            scorers_nms=["action_dist", "loss", "weight_res_mean", 
                         "weight_res_std"]
            )
        self.importance_sampler = importance_sampler
        self.norm_weights = norm_weights
        self.clip = clip
        self.discount = discount
        self.behav_policy = behav_policy
        self.episodes = episodes
        self.unique_pol_acts = unique_pol_acts
        self.is_kwargs = is_kwargs
        self.gpu = gpu
        
    def eval(self, algo: AlgoProtocol, epoch, total_step) -> Dict:
        policy_class = D3RlPyTorchAlgoPredict(
            predict_func=algo.predict)
        
        eval_policy = D3RlPyDeterministic(policy_class=policy_class, 
                                          collect_res=False, collect_act=True,
                                          gpu=self.gpu)
        
        is_calculator = self.importance_sampler(
            behav_policy=self.behav_policy, eval_policy=eval_policy, 
            discount=self.discount, **self.is_kwargs)

        weight_res = []
        discnt_reward_res = []
        norm_conts = []
        # For each trajectory in an input dataset, get a list of the weights
        # and rewards for each trajectory
        # TODO: Consolodate with is_pipeline code!
        for episode in self.episodes:
            weight, discnt_reward, norm_cont = is_calculator.get_traj_w_r(
                state=torch.Tensor(episode.observations), 
                action=torch.Tensor(episode.actions.reshape(-1,1)), 
                reward=torch.Tensor(episode.rewards.reshape(-1,1))
                )
            weight_res.append(weight)
            discnt_reward_res.append(discnt_reward)
            norm_conts.append(norm_cont)

        if self.norm_weights:
            norm_val = torch.sum(torch.Tensor(norm_conts))
            weight_res = [w/norm_val for w in weight_res]
        else:
            weight_res = [w/len(self.episodes) for w in weight_res]
        
        weight_res = torch.concat(weight_res)
        discnt_reward_res = torch.concat(discnt_reward_res)
    
        loss, _, clip_loss, _, weight_res, clip_weight_res = eval_weight_array(
            weight_res=weight_res, discnt_reward_res=discnt_reward_res,
            save_dir=None, prefix=None, clip=self.clip)
        # TODO: Assumes 1 dimensional action!
        eval_policy_acts = [tens.squeeze().detach().numpy().reshape(-1) 
                            for tens in eval_policy.policy_actions]
        eval_policy_acts = np.concatenate(eval_policy_acts)
        action_dist = {}
        for i in self.unique_pol_acts:
            action_dist[i] = (
                (eval_policy_acts == i).sum()/len(eval_policy_acts)
            )
        res = {
            "action_dist": action_dist,
            "loss": loss,
            "weight_res_mean": weight_res.mean(), 
            "weight_res_std": weight_res.std()
        }
        return res

class ActDistCache(MultiOutputCache):
    def __init__(self, unique_action_vals:List, 
                 eval_wrap:TorchISEvalD3rlpyWrap) -> None:
        super().__init__(unique_values=unique_action_vals)
        self.eval_wrap = eval_wrap
    
    def scoring_calc(self, algo: AlgoProtocol, episodes: List[Episode]):
        return self.eval_wrap["action_dist"]        
 