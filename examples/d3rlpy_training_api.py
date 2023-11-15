from d3rlpy.algos import DQNConfig
from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics import evaluate_qlearning_with_environment
from d3rlpy.metrics import (SoftOPCEvaluator, 
                            InitialStateValueEstimationEvaluator)
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.dataset import BasicTransitionPicker
from d3rlpy.ope import DiscreteFQE
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import shutil

# Import callbacks
from offline_rl_ope.api.d3rlpy.Callbacks import (
    
    ISCallback, FQECallback, EpochCallbackHandler, 
    DiscreteValueByActionCallback
    )

# Import evaluators
from offline_rl_ope.api.d3rlpy.Scorers import (
    ISEstimatorScorer, ISDiscreteActionDistScorer, QueryScorer
    )
from offline_rl_ope.components.Policy import BehavPolicy

# obtain dataset
dataset, env = get_cartpole()

# setup algorithm
gamma = 0.99
dqn = DQNConfig(gamma=gamma, target_update_interval=100).create()

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
observations = []
actions = []
tp = BasicTransitionPicker()
for ep in dataset.episodes:
    for i in range(ep.transition_count):
        _transition = tp(ep,i)
        observations.append(_transition.observation.reshape(1,-1))
        actions.append(_transition.action)

observations = np.concatenate(observations)
actions = np.concatenate(actions)

behav_est.fit(X=observations, Y=actions.reshape(-1,1))

gbt_est = GbtEst(estimator=behav_est)
gbt_policy_be = BehavPolicy(policy_class=gbt_est, collect_res=False)


is_callback = ISCallback(
    is_types=["vanilla", "per_decision"], 
    behav_policy=gbt_policy_be, 
    dataset=dataset,
    eval_policy_kwargs={
        "gpu": False, 
        "collect_act": True   
    }
    )

fqe_scorers = {
    "soft_opc": SoftOPCEvaluator(70), 
    "init_state_val": InitialStateValueEstimationEvaluator()
}

fqe_init_kwargs = {
    "q_func_factory": MeanQFunctionFactory(), "learning_rate": 1e-4
    }

fqe_fit_kwargs = {"n_steps":len(actions)}


fqe_callback = FQECallback(
    scorers=fqe_scorers, fqe_cls=DiscreteFQE, model_init_kwargs=fqe_init_kwargs, 
    model_fit_kwargs=fqe_fit_kwargs, dataset=dataset)


dva_callback = DiscreteValueByActionCallback(
    unique_action_vals=unique_pol_acts, dataset=dataset)


scorers = {}

scorers.update({"vanilla_is_loss": ISEstimatorScorer(
    discount=gamma, cache=is_callback, is_type="vanilla", norm_weights=False)})

scorers.update({"pd_is_loss": ISEstimatorScorer(
    discount=gamma, cache=is_callback, is_type="per_decision", 
    norm_weights=False)})

scorers.update({"vanilla_wis_loss": ISEstimatorScorer(
    discount=gamma, cache=is_callback, is_type="vanilla", norm_weights=True)})

scorers.update({"vanilla_wis_loss_smooth": ISEstimatorScorer(
    discount=gamma, cache=is_callback, is_type="vanilla", norm_weights=True, 
    norm_kwargs={"smooth_eps":0.0000001})})

scorers.update({"pd_wis_loss": ISEstimatorScorer(
    discount=gamma, cache=is_callback, is_type="per_decision", 
    norm_weights=True)})

for act in unique_pol_acts:
    scorers.update(
        {
            "action_dist_{}".format(act): ISDiscreteActionDistScorer(
                cache=is_callback, act=act)
            }
        )
    scorers.update(
        {
            "action_value_{}".format(act): QueryScorer(
                cache=dva_callback, query_key=act)
            }
        )
    
for scr in fqe_scorers:
    scorers.update({scr: QueryScorer(cache=fqe_callback, query_key=scr)})

epoch_callback = EpochCallbackHandler([is_callback, fqe_callback, dva_callback])

n_epochs=2
n_steps_per_epoch = len(actions)
n_steps = len(actions)*n_epochs
dqn.fit(dataset, n_steps=n_steps, n_steps_per_epoch=n_steps_per_epoch, 
        evaluators=scorers, epoch_callback=epoch_callback, 
        with_timestamp=False)

# The FQE model dumps results to a temporary location. Clean this up on close!
fqe_callback.clean_up()

# evaluate trained algorithm
evaluate_qlearning_with_environment(algo=dqn, env=env)
shutil.rmtree("d3rlpy_data")
shutil.rmtree("d3rlpy_logs")