from d3rlpy.algos import DQNConfig
import pandas as pd
from d3rlpy.datasets import get_cartpole
import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from d3rlpy.ope import DiscreteFQE, FQEConfig
from d3rlpy.metrics import (SoftOPCEvaluator, 
                            InitialStateValueEstimationEvaluator)
from d3rlpy.dataset import BasicTransitionPicker, ReplayBuffer, InfiniteBuffer
from xgboost import XGBClassifier
import torch

from offline_rl_ope.Dataset import ISEpisode
from offline_rl_ope.components.Policy import BehavPolicy, GreedyDeterministic
from offline_rl_ope.components.ImportanceSampler import ISWeightOrchestrator
from offline_rl_ope.OPEEstimators import (
    ISEstimator, DREstimator, D3rlpyQlearnDM)
from offline_rl_ope.PropensityModels.sklearn import (
    MultiOutputMultiClassTrainer, SklearnTorchTrainerWrapper)
from offline_rl_ope.LowerBounds.HCOPE import get_lower_bound

from offline_rl_ope.api.d3rlpy.Misc import D3RlPyTorchAlgoPredict

if __name__ == "__main__":
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

    sklearn_trainer = MultiOutputMultiClassTrainer(
        theoretical_action_classes=[np.array([0,1])],
        estimator=behav_est
        )
    sklearn_trainer.fitted_cls = [pd.Series(actions).unique()]
    gbt_est = SklearnTorchTrainerWrapper(
        sklearn_trainer=sklearn_trainer
    )
    gbt_policy_be = BehavPolicy(
        policy_func=gbt_est, 
        collect_res=False
        )

    no_obs_steps = int(len(actions)*0.025)
    n_epochs=1
    n_steps_per_epoch = no_obs_steps
    n_steps = no_obs_steps*n_epochs
    dqn.fit(dataset, n_steps=n_steps, n_steps_per_epoch=n_steps_per_epoch, 
            with_timestamp=False)

    fqe_scorers = {
        "soft_opc": SoftOPCEvaluator(
            return_threshold=70,
            episodes=dataset.episodes
            ),
        "init_state_val": InitialStateValueEstimationEvaluator(
            episodes=dataset.episodes
        )
    }


    fqe_config = FQEConfig(learning_rate=1e-4)
    #discrete_fqe = DiscreteFQE(algo=dqn, **fqe_init_kwargs)
    discrete_fqe = DiscreteFQE(algo=dqn, config=fqe_config, device=False)

    discrete_fqe.fit(dataset, evaluators=fqe_scorers, n_steps=no_obs_steps)            

        
    # Static OPE evaluation 
    policy_func = D3RlPyTorchAlgoPredict(predict_func=dqn.predict)
    eval_policy = GreedyDeterministic(
        policy_func=policy_func, collect_res=False, 
        collect_act=True, gpu=False
        )

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
    wis_estimator_smooth = ISEstimator(norm_weights=True, norm_kwargs={
        "smooth_eps":0.0000001
    })
    w_dr_estimator = DREstimator(
        dm_model=fqe_dm_model, norm_weights=True, 
                                ignore_nan=True)


    res = is_estimator.predict(
        rewards=[ep.reward for ep in episodes], discount=0.99,
        weights=is_weight_calculator["vanilla"].traj_is_weights, 
        is_msk=is_weight_calculator.weight_msk, 
        states=[ep.state for ep in episodes], 
        actions=[ep.action for ep in episodes],
    )
    print(res)

    res = is_estimator.predict(
        weights=is_weight_calculator["per_decision"].traj_is_weights, 
        rewards=[ep.reward for ep in episodes], discount=0.99,
        is_msk=is_weight_calculator.weight_msk, 
        states=[ep.state for ep in episodes], 
        actions=[ep.action for ep in episodes],
    )
    print(res)
    traj_rewards = is_estimator.traj_rewards_cache.squeeze().numpy()
    print(get_lower_bound(X=traj_rewards, delta=0.05))

    res = wis_estimator.predict(
        rewards=[ep.reward for ep in episodes], discount=0.99,
        weights=is_weight_calculator["vanilla"].traj_is_weights, 
        is_msk=is_weight_calculator.weight_msk, 
        states=[ep.state for ep in episodes], 
        actions=[ep.action for ep in episodes]
        )
    print(res)

    res = wis_estimator.predict(
        weights=is_weight_calculator["per_decision"].traj_is_weights, 
        rewards=[ep.reward for ep in episodes], discount=0.99,
        is_msk=is_weight_calculator.weight_msk, 
        states=[ep.state for ep in episodes], 
        actions=[ep.action for ep in episodes],
    )
    print(res)

    res = wis_estimator_smooth.predict(
        weights=is_weight_calculator["vanilla"].traj_is_weights, 
        rewards=[ep.reward for ep in episodes], discount=0.99,
        is_msk=is_weight_calculator.weight_msk, 
        states=[ep.state for ep in episodes], 
        actions=[ep.action for ep in episodes],
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

