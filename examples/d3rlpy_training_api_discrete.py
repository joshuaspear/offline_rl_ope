from d3rlpy.algos import DQNConfig
from d3rlpy.datasets import get_cartpole
from d3rlpy.metrics import evaluate_qlearning_with_environment
from d3rlpy.metrics import (SoftOPCEvaluator, 
                            InitialStateValueEstimationEvaluator)
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.dataset import BasicTransitionPicker, ReplayBuffer, InfiniteBuffer
from d3rlpy.ope import DiscreteFQE
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from xgboost import XGBClassifier
import pandas as pd
import shutil

from offline_rl_ope.api.d3rlpy.Policy import PolicyFactory

# Import callbacks
from offline_rl_ope.api.d3rlpy.Callbacks import (
    ISCallback, FQECallback, EpochCallbackHandler, 
    DiscreteValueByActionCallback
    )

# Import evaluators
from offline_rl_ope.api.d3rlpy.Scorers import (
    ISEstimatorScorer, ISDiscreteActionDistScorer, QueryScorer
    )
from offline_rl_ope.components.Policy import NumpyPolicyFuncWrapper, Policy
from offline_rl_ope.PropensityModels.sklearn import (
    SklearnDiscrete)

if __name__=="__main__":

    # obtain dataset
    dataset, env = get_cartpole()

    # setup algorithm
    gamma = 0.99
    dqn = DQNConfig(gamma=gamma, target_update_interval=100).create()

    unique_pol_acts = np.arange(0,env.action_space.n)

    behav_est = MultiOutputClassifier(OneVsRestClassifier(XGBClassifier(
        objective="binary:logistic")))
    dataset = ReplayBuffer(
        buffer=InfiniteBuffer(), 
        episodes=dataset.episodes[0:100]
        ) 

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

    sklearn_trainer = SklearnDiscrete(
        theoretical_action_classes=[np.array([0,1])],
        estimator=behav_est
        )
    sklearn_trainer.fitted_cls = [pd.Series(actions).unique()]

    gbt_policy_be = Policy(
        policy_func=NumpyPolicyFuncWrapper(sklearn_trainer.policy_func), 
        collect_res=False
        )


    is_callback = ISCallback(
        is_types=["vanilla", "per_decision"], 
        behav_policy=gbt_policy_be, 
        dataset=dataset,
        policy_factory=PolicyFactory(
            deterministic=True,
            collect_res=False,
            collect_act=True,
            gpu=False,
            action_dim=1
        )
        )

    fqe_scorers = {
        "soft_opc": SoftOPCEvaluator(70), 
        "init_state_val": InitialStateValueEstimationEvaluator()
    }

    fqe_init_kwargs = {
        "q_func_factory": MeanQFunctionFactory(), "learning_rate": 1e-4
        }

    fqe_fit_kwargs = {
        "n_steps":len(actions),
        "n_steps_per_epoch":len(actions)
        }


    fqe_callback = FQECallback(
        scorers=fqe_scorers, fqe_cls=DiscreteFQE, 
        model_init_kwargs=fqe_init_kwargs, 
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
        
    # for scr in fqe_scorers:
    #     scorers.update({scr: QueryScorer(cache=fqe_callback, query_key=scr)})

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