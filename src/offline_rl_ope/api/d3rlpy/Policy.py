from typing import Optional, Union
from d3rlpy.algos.qlearning.base import QLearningAlgoBase

from jaxtyping import jaxtyped
from typeguard import typechecked as typechecker

from .Misc import (
    D3RlPyDeterministicDiscreteWrapper, D3RlPyStochasticWrapper
    )
from ...components.Policy import (
    Policy, GreedyDeterministic)

__all__ = ["PolicyFactory"]
            
class PolicyFactory:
    
    def __init__(
        self,
        deterministic:bool,
        collect_res:bool=False, 
        collect_act:bool=False, 
        gpu:bool=False,
        action_dim:Optional[int]=None
        ):
        self.deterministic = deterministic
        self.eval_policy_kwargs = {
            "collect_res": collect_res,
            "collect_act": collect_act,
            "gpu": gpu
        }
        if self.deterministic:
            assert action_dim is not None
        self.action_dim = action_dim
        
    def __create_deterministic(self, algo:QLearningAlgoBase):
        policy_func = D3RlPyDeterministicDiscreteWrapper(
            predict_func=algo.predict,
            action_dim=self.action_dim
            )
        eval_policy = GreedyDeterministic(
            policy_func=policy_func, 
            **self.eval_policy_kwargs
            )
        return eval_policy
        
    def __create_stochastic(self, algo:QLearningAlgoBase):
        policy_func = D3RlPyStochasticWrapper(
            policy_func=algo.impl.policy,
            observation_scaler=algo._config.observation_scaler,
            action_scaler=algo._config.action_scaler
            )
    
        eval_policy = Policy(
            policy_func=policy_func, 
            **self.eval_policy_kwargs
        )
        return eval_policy
    
    @jaxtyped(typechecker=typechecker)
    def create(self, algo: QLearningAlgoBase)->Policy:
        if self.deterministic:
            res = self.__create_deterministic(algo=algo)
        else:
            res = self.__create_stochastic(algo=algo)
        return res