from d3rlpy.algos import SACConfig
from d3rlpy.datasets import get_pendulum
from typing import Dict
from d3rlpy.ope import FQEConfig, FQE
from d3rlpy.metrics import (SoftOPCEvaluator, 
                            InitialStateValueEstimationEvaluator)
from d3rlpy.dataset import BasicTransitionPicker, ReplayBuffer, InfiniteBuffer
import numpy as np
import torch
from torch.distributions import Normal

from pymlrf.SupervisedLearning.torch import (
    PercEpsImprove)
from pymlrf.FileSystem import DirectoryHandler

from offline_rl_ope.Dataset import ISEpisode
from offline_rl_ope.components.Policy import Policy, GreedyDeterministic
from offline_rl_ope.components.ImportanceSampler import ISWeightOrchestrator
from offline_rl_ope.OPEEstimators import (
    ISEstimator, DREstimator, D3rlpyQlearnDM)
from offline_rl_ope.PropensityModels.torch import FullGuassian, TorchRegTrainer 
from offline_rl_ope.LowerBounds.HCOPE import get_lower_bound

from offline_rl_ope.api.d3rlpy.Misc import D3RlPyTorchAlgoPredict
from offline_rl_ope.types import PropensityTorchOutputType

from PropensityTrainingLoop import torch_training_loop

class GaussianLossWrapper:
    
    def __init__(self) -> None:
        self.scorer = torch.nn.GaussianNLLLoss()
    
    def __call__(
        self, 
        y_pred:PropensityTorchOutputType, 
        y_true:Dict[str,torch.Tensor]
        ) -> torch.Tensor:
        res = self.scorer(
            input=y_pred["loc"], 
            var=y_pred["scale"], 
            target=y_true["y"]
            )
        return res

if __name__ == "__main__":
    # obtain dataset
    dataset, env = get_pendulum()

    # setup algorithm
    gamma = 0.99
    sac = SACConfig(gamma=gamma).create()
    
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

    assert len(env.observation_space.shape) == 1
    estimator = FullGuassian(
            input_dim=env.observation_space.shape[0], 
            #layers_dim=[64,64],
            layers_dim=[64,64],
            m_out_dim=1, 
            sd_out_dim=1
            )
    estimator = TorchRegTrainer(
        estimator=estimator, 
        dist_func=Normal, 
        gpu=False
        )
    early_stop_criteria = PercEpsImprove(eps=0, direction="gr")
    meta_data = {
        "train_loss_criteria": "gauss_nll",
        "val_loss_criteria": "gauss_nll"
        }
    criterion = GaussianLossWrapper()
    
    prop_output_dh = DirectoryHandler(loc="./propensity_output")
    if not prop_output_dh.is_created:
        prop_output_dh.create()
    else:
        prop_output_dh.clear()
    
    torch_training_loop.train(
        model=estimator.estimator,
        x_train=observations,
        y_train=actions.reshape(-1,1),
        x_val=observations,
        y_val=actions.reshape(-1,1),
        batch_size=32,
        shuffle=True,
        lr=0.01,
        gpu=False,
        criterion=criterion,
        epochs=4,
        seed=1,
        save_dir=prop_output_dh.loc,
        early_stopping_func=early_stop_criteria
    )
    
    policy_be = Policy(
        policy_func=estimator.predict_proba, 
        collect_res=False
        )

    no_obs_steps = int(len(actions)*0.025)
    n_epochs=1
    n_steps_per_epoch = no_obs_steps
    n_steps = no_obs_steps*n_epochs
    sac.fit(
        dataset, n_steps=n_steps, n_steps_per_epoch=n_steps_per_epoch, 
        with_timestamp=False
        )

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
    fqe = FQE(algo=sac, config=fqe_config, device=False)

    fqe.fit(dataset, evaluators=fqe_scorers, n_steps=no_obs_steps)            

        
    # Static OPE evaluation 
    policy_func = D3RlPyTorchAlgoPredict(
        predict_func=sac.predict,
        action_dim=1
        )
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
                                                behav_policy=policy_be)
    is_weight_calculator.update(
        states=[ep.state for ep in episodes], 
        actions=[ep.action for ep in episodes],
        eval_policy=eval_policy)

    fqe_dm_model = D3rlpyQlearnDM(model=fqe)

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

