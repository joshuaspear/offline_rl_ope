from d3rlpy.algos import SACConfig
from d3rlpy.datasets import get_pendulum
from d3rlpy.metrics import evaluate_qlearning_with_environment
from d3rlpy.metrics import (SoftOPCEvaluator, 
                            InitialStateValueEstimationEvaluator)
from d3rlpy.models.q_functions import MeanQFunctionFactory
from d3rlpy.dataset import BasicTransitionPicker, ReplayBuffer, InfiniteBuffer
from d3rlpy.ope import FQE
import numpy as np
import shutil
from torch.distributions import Normal


from pymlrf.SupervisedLearning.torch import (
    PercEpsImprove)
from pymlrf.FileSystem import DirectoryHandler



from PropensityTrainingLoop import torch_training_loop, GaussianLossWrapper

from offline_rl_ope.PropensityModels.torch import FullGuassian, TorchRegTrainer 
from offline_rl_ope.api.d3rlpy.Policy import PolicyFactory

# Import callbacks
from offline_rl_ope.api.d3rlpy.Callbacks import (
    ISCallback, FQECallback, EpochCallbackHandler
    )

# Import evaluators
from offline_rl_ope.api.d3rlpy.Scorers import (
    ISEstimatorScorer, QueryScorer
    )
from offline_rl_ope.components.Policy import Policy

if __name__=="__main__":

    # obtain dataset
    dataset, env = get_pendulum()

    # setup algorithm
    gamma = 0.99
    sac = SACConfig(gamma=gamma).create()

    dataset = ReplayBuffer(
        buffer=InfiniteBuffer(), 
        episodes=dataset.episodes[0:5]
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
        policy_func=estimator.policy_func, 
        collect_res=False
        )


    is_callback = ISCallback(
        is_types=["vanilla", "per_decision"], 
        behav_policy=policy_be, 
        dataset=dataset,
        policy_factory=PolicyFactory(
            deterministic=False,
            collect_res=False,
            collect_act=True,
            gpu=False
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
        scorers=fqe_scorers, fqe_cls=FQE, 
        model_init_kwargs=fqe_init_kwargs, 
        model_fit_kwargs=fqe_fit_kwargs, dataset=dataset)

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
        
    for scr in fqe_scorers:
        scorers.update({scr: QueryScorer(cache=fqe_callback, query_key=scr)})

    epoch_callback = EpochCallbackHandler([is_callback, fqe_callback])

    n_epochs=2
    n_steps_per_epoch = len(actions)
    n_steps = len(actions)*n_epochs
    sac.fit(dataset, n_steps=n_steps, n_steps_per_epoch=n_steps_per_epoch, 
            evaluators=scorers, epoch_callback=epoch_callback, 
            with_timestamp=False)

    # The FQE model dumps results to a temporary location. Clean this up on close!
    fqe_callback.clean_up()

    # evaluate trained algorithm
    evaluate_qlearning_with_environment(algo=sac, env=env)
    shutil.rmtree("d3rlpy_data")
    shutil.rmtree("d3rlpy_logs")