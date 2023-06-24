from d3rlpy.algos import DQN
from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics.scorer import evaluate_on_environment
from d3rlpy.metrics.scorer import (soft_opc_scorer, 
    initial_state_value_estimation_scorer)
from d3rlpy.ope import DiscreteFQE
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier

from offline_rl_ope.api.d3rlpy import (
    TorchISEvalD3rlpyWrap, FqeEvalD3rlpyWrap, WrapperAccessor, 
    EpochCallbackHandler)
from offline_rl_ope.api.d3rlpy.scorers import (
    DiscreteValueByActionCache)
from offline_rl_ope.components.ImportanceSampler import VanillaIS
from offline_rl_ope.components.Policy import BehavPolicy
# from offline_rl_ope.api.d3rlpy.is_evaluation import ActDistCache
# from offline_rl_ope.components.utils import MultiOutputScorer

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

# Define an importance sampling calculation and caching object using weighted 
# vanilla importance sampling, without clipping weights
eval_wrap = TorchISEvalD3rlpyWrap(
    importance_sampler=VanillaIS, behav_policy=gbt_policy_be,
    discount=gamma, episodes=dataset.episodes, norm_weights=True, clip=None, 
    unique_pol_acts=unique_pol_acts)

# Define caching object for capturing the average value by action, according to 
# the evaluation model across the evaluation set
disc_val_by_act_cache = DiscreteValueByActionCache(
    unique_action_vals=list(unique_pol_acts))

# Define caching object for capturing the distribution of actions under the 
# evaluation policy on the validation set
act_dist_cache = ActDistCache(
    unique_action_vals=list(unique_pol_acts), 
    eval_wrap=eval_wrap)


scorers = {}

# Define scorer to extract the weighted IS loss from the TorchISEvalD3rlpyWrap 
# object
scorers.update({"loss": WrapperAccessor(item_key="loss", wrapper=eval_wrap)})

# Define scorer to extract the mean IS weights from the TorchISEvalD3rlpyWrap 
# object
scorers.update({"weight_res_mean": WrapperAccessor(
    item_key="weight_res_mean", wrapper=eval_wrap)})

# Define scorer to extract the standard deviation of IS weights from the 
# TorchISEvalD3rlpyWrap object
scorers.update({"weight_res_std": WrapperAccessor(
    item_key="weight_res_std", wrapper=eval_wrap)})

# Define a scorer per action to extract the value from the 
# DiscreteValueByActionCache object
for i in unique_pol_acts:
    __scor_nm = "value_by_action_{}".format(i)
    scorers.update({
        __scor_nm: MultiOutputScorer(value=i, cache=disc_val_by_act_cache)
        })

# Define a scorer per action to extract the probability of the action from the  
# ActDistCache object during the IS evaluation
for i in unique_pol_acts:
    __scor_nm = "act_dist_{}".format(i)
    scorers.update({
        __scor_nm: MultiOutputScorer(value=i, cache=act_dist_cache)
        })

# Define the scorers and other hyperparameters required for the FQE evaluation
fqe_scorers = {
    "soft_opc": soft_opc_scorer(70), 
    "init_state_val": initial_state_value_estimation_scorer
}

fqe_init_kwargs = {"use_gpu": False, "discrete_action": True, 
                   "q_func_factory": 'mean', "learning_rate": 1e-4
                   }

fqe_fit_kwargs = {"n_epochs":1}

fqe_eval_wrap = FqeEvalD3rlpyWrap(
    fqe_cls=DiscreteFQE, scorers=fqe_scorers, 
    model_init_kwargs=fqe_init_kwargs, model_fit_kwargs=fqe_fit_kwargs, 
    dataset=dataset)

for scr in fqe_scorers.keys():
    scorers.update({scr:WrapperAccessor(
        item_key=scr, wrapper=fqe_eval_wrap)})

# Combine both the IS and FQE calculation/caching objects into a single callback
epoch_callback = EpochCallbackHandler([eval_wrap, fqe_eval_wrap])

# Train:
# After each epoch: 
# * Regression weighted importance sampling will be performed and the loss, 
#   mean IS weights, std IS weights will be recorded as well as the probability 
#   of each action in the IS evaluation
# * An FQE model will be trained for a single epoch and the soft OPC and 
#   mean initial state value according to the trained FQE model
# * The mean value according to the evaluation model on the evaluation set, 
#   split by action
dqn.fit(dataset.episodes, n_epochs=5, scorers=scorers, 
        eval_episodes=dataset.episodes, epoch_callback=epoch_callback)

# The FQE model dumps results to a temporary location. Clean this up on close!
fqe_eval_wrap.clean_up()

# evaluate trained algorithm
evaluate_on_environment(env, render=True)(dqn)