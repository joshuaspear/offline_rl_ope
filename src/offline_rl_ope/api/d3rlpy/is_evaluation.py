import numpy as np
import torch
from typing import List, Callable
from d3rlpy.metrics.scorer import AlgoProtocol
from d3rlpy.dataset import Episode

from ...is_eval_base import eval_weight_array
from ...components.Policy import Policy, D3RlPyDeterministic
from ...components.ImportanceSampling import ImportanceSampling

class D3RlPyTorchAlgoPredict:
    
    def __init__(self, predict_func:Callable):
        self.predict_func = predict_func
        
    def __call__(self, x):
        return torch.Tensor(self.predict_func(x))    
 
class TorchISEvalD3rlpyWrap:

    def __init__(self, importance_sampler:ImportanceSampling,
                mixing_params:List[float], discount:float, 
                behav_policy:Policy, state_size:int, action_size:int, 
                episodes: List[Episode], norm_weights:bool=False, 
                clip:float=None, is_kwargs={}, unique_pol_acts = [0,1]
                ) -> None:
        self.importance_sampler = importance_sampler
        self.norm_weights = norm_weights
        self.clip = clip
        self.discount = discount
        self.mixing_params = mixing_params
        self.behav_policy = behav_policy
        self.state_size = state_size
        self.action_size = action_size
        self.episodes = episodes
        self.unique_pol_acts = unique_pol_acts
        self.loss = None
        self.weight_res_mean = None
        self.weight_res_std = None
        self.no_presc = None
        self.is_kwargs = is_kwargs
    
    
    def __flush(self):
        
        self.loss = None
        self.weight_res_mean = None
        self.weight_res_std = None
        self.no_presc = None

        
        
    def __call__(self, algo: AlgoProtocol, epoch, total_step):
        self.__flush()
        policy_class = D3RlPyTorchAlgoPredict(
            predict_func=algo.predict)
        
        eval_policy = D3RlPyDeterministic(policy_class=policy_class, 
                                          collect_res=False, collect_act=True)
        
        is_calculator = self.importance_sampler(
            behav_policy=self.behav_policy, eval_policy=eval_policy, 
            discount=self.discount, **self.is_kwargs)

        weight_res = []
        discnt_reward_res = []
        for episode in self.episodes:
            #for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            weight, discnt_reward = is_calculator.get_traj_loss(
                state=torch.Tensor(episode.observations), 
                action=torch.Tensor(episode.actions.reshape(-1,1)), 
                reward=torch.Tensor(episode.rewards.reshape(-1,1)))
            weight_res.append(weight)
            discnt_reward_res.append(discnt_reward)
        weight_res = np.array(weight_res)
        discnt_reward_res = np.array(discnt_reward_res)
        loss, _, clip_loss, _, weight_res, clip_weight_res = eval_weight_array(
            weight_res=weight_res, discnt_reward_res=discnt_reward_res, 
            dataset_len=len(self.episodes), norm_weights=self.norm_weights, 
            save_dir=None, prefix=None, clip=self.clip)
        # TODO: Assumes 1 dimensional action!
        eval_policy_acts = [tens.squeeze().detach().numpy().reshape(-1) 
                            for tens in eval_policy.policy_actions]
        eval_policy_acts = np.concatenate(eval_policy_acts)
        no_presc_res = []
        for i in self.unique_pol_acts:
            no_presc_res.append(
                (eval_policy_acts == 1).sum()/len(eval_policy_acts)
                )
        self.no_presc = np.array(no_presc_res)
        self.loss = loss
        self.weight_res_mean = weight_res.mean()
        self.weight_res_std = weight_res.std()

            
class d3rlpy_loss_scorer:
    
    def __init__(self, eval_wrap:TorchISEvalD3rlpyWrap) -> None:
        self.eval_wrap = eval_wrap
    
    def __call__(self, algo: AlgoProtocol, episodes: List[Episode]):
        return self.eval_wrap.loss
    
    
class d3rlpy_weight_mean_scorer:
    
    def __init__(self, eval_wrap:TorchISEvalD3rlpyWrap) -> None:
        self.eval_wrap = eval_wrap
    
    def __call__(self, algo: AlgoProtocol, episodes: List[Episode]):
        return self.eval_wrap.weight_res_mean
    
    
class d3rlpy_weight_std_scorer:
    
    def __init__(self, eval_wrap:TorchISEvalD3rlpyWrap) -> None:
        self.eval_wrap = eval_wrap
    
    def __call__(self, algo: AlgoProtocol, episodes: List[Episode]):
        return self.eval_wrap.weight_res_std
    

class d3rlpy_no_presc_scorer:
    
    def __init__(self, eval_wrap:TorchISEvalD3rlpyWrap) -> None:
        self.eval_wrap = eval_wrap
    
    def __call__(self, algo: AlgoProtocol, episodes: List[Episode]):
        return self.eval_wrap.no_presc
