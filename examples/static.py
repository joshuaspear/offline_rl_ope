from d3rlpy.algos import DQN
import pickle
from d3rlpy.datasets import get_cartpole
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from d3rlpy.ope import DiscreteFQE
from d3rlpy.metrics.scorer import (soft_opc_scorer, 
    initial_state_value_estimation_scorer)
from xgboost import XGBClassifier
import math
import torch

from offline_rl_ope.Dataset import ISEpisode
from offline_rl_ope.components.Policy import BehavPolicy, D3RlPyDeterministic
from offline_rl_ope.components.ImportanceSampler import ISWeightOrchestrator
from offline_rl_ope.OPEEstimators import (
    ISEstimator, DREstimator, D3rlpyQlearnDM)
from offline_rl_ope.LowerBounds.HCOPE import get_lower_bound

from offline_rl_ope.api.d3rlpy.ISScorer import D3RlPyTorchAlgoPredict

# obtain dataset
dataset, env = get_cartpole()

# setup algorithm
gamma = 0.99
dqn = DQN(gamma=gamma, target_update_interval=100)

unique_pol_acts = np.arange(0,env.action_space.n)

# Class for handling the output of the XGBoost classifier for regression based 
# importance sampling. Can handle arbitrary numbers of classes and dimensions 
# of discrete actions.

class GbtEst:
    
    def __init__(self, estimator:MultiOutputClassifier) -> None:
        self.estimator = estimator
    
    def eval_pdf(self, indep_vals:np.array, dep_vals:np.array):
        probs = self.estimator.predict_proba(X=indep_vals)
        res = []
        for i,out_prob in enumerate(probs):
            tmp_res = out_prob[
                np.arange(len(out_prob)),
                dep_vals[:,i].squeeze().astype(int)
                ]
            res.append(tmp_res.reshape(1,-1))
        res = np.concatenate(res, axis=0).prod(axis=0)
        return res

behav_est = MultiOutputClassifier(OneVsRestClassifier(XGBClassifier(
    objective="binary:logistic")))

# Fit the behaviour model
behav_est.fit(X=dataset.observations, Y=dataset.actions.reshape(-1,1))

gbt_est = GbtEst(estimator=behav_est)
gbt_policy_be = BehavPolicy(policy_class=gbt_est, collect_res=False)

dqn.fit(dataset.episodes, n_epochs=1)

fqe_scorers = {
    "soft_opc": soft_opc_scorer(70), 
    "init_state_val": initial_state_value_estimation_scorer
}

fqe_init_kwargs = {"use_gpu": False, "discrete_action": True, 
                   "q_func_factory": 'mean', "learning_rate": 1e-4
                  }
discrete_fqe = DiscreteFQE(algo=dqn, **fqe_init_kwargs)

discrete_fqe.fit(dataset.episodes, eval_episodes=dataset.episodes, 
                    scorers=fqe_scorers, n_epochs=1)            

     
# Static OPE evaluation 
policy_class = D3RlPyTorchAlgoPredict(predict_func=dqn.predict)
eval_policy = D3RlPyDeterministic(policy_class=policy_class, collect_res=False, 
                                  collect_act=True, gpu=False)

episodes = []
for ep in dataset.episodes:
    episodes.append(ISEpisode(
        state=torch.Tensor(ep.observations), 
        action=torch.Tensor(ep.actions).view(-1,1),
        reward=torch.Tensor(ep.rewards))
                    )
    
is_weight_calculator = ISWeightOrchestrator("vanilla", "per_decision", 
                                            behav_policy=gbt_policy_be)
is_weight_calculator.update(
    states=[ep.state for ep in episodes], 
    actions=[ep.action for ep in episodes],
    eval_policy=eval_policy)

fqe_dm_model = D3rlpyQlearnDM(model=discrete_fqe)

is_estimator = ISEstimator(norm_weights=False, cache_traj_rewards=True)
wis_estimator = ISEstimator(norm_weights=True)
w_dr_estimator = DREstimator(dm_model=fqe_dm_model, norm_weights=True, 
                             ignore_nan=True)


res = is_estimator.predict(
    rewards=[ep.reward for ep in episodes], discount=0.99,
    weights=is_weight_calculator["vanilla"].traj_is_weights, 
    is_msk=is_weight_calculator.weight_msk, states=[], actions=[]
)
print(res)

res = is_estimator.predict(
    weights=is_weight_calculator["per_decision"].traj_is_weights, 
    rewards=[ep.reward for ep in episodes], discount=0.99,
    is_msk=is_weight_calculator.weight_msk, states=[], actions=[]
)
print(res)
traj_rewards = is_estimator.traj_rewards_cache.squeeze().numpy()
print(get_lower_bound(X=traj_rewards, delta=0.05))

res = wis_estimator.predict(
    rewards=[ep.reward for ep in episodes], discount=0.99,
    weights=is_weight_calculator["vanilla"].traj_is_weights, 
    is_msk=is_weight_calculator.weight_msk, states=[], actions=[])
print(res)

res = wis_estimator.predict(
    weights=is_weight_calculator["per_decision"].traj_is_weights, 
    rewards=[ep.reward for ep in episodes], discount=0.99,
    is_msk=is_weight_calculator.weight_msk, states=[], actions=[]
)
print(res)

res = w_dr_estimator.predict(
    weights=is_weight_calculator["per_decision"].traj_is_weights, 
    rewards=[ep.reward for ep in episodes], discount=0.99,
    is_msk=is_weight_calculator.weight_msk, 
    states=[ep.state for ep in episodes], 
    actions=[ep.action for ep in episodes],
)
print(res)

