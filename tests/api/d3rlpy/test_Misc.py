import unittest
from unittest.mock import MagicMock
from ...d3rlpy_test_utils import init_trained_algo, create_observations
import numpy as np
from numpy.testing import assert_allclose
from parameterized import parameterized_class
from d3rlpy.algos.qlearning import SACConfig, DQNConfig
from d3rlpy.models.torch.policies import ActionOutput
from d3rlpy.preprocessing.action_scalers import MinMaxActionScaler
from d3rlpy.models.torch.policies import build_squashed_gaussian_distribution
from d3rlpy.preprocessing.observation_scalers import StandardObservationScaler
from offline_rl_ope.api.d3rlpy.Misc import (
    D3RlPyDeterministicDiscreteWrapper, D3RlPyStochasticWrapper
    )
import torch
from d3rlpy.dataset import (
    EpisodeGenerator,
    create_infinite_replay_buffer,
)
import numpy.typing as npt
Float32NDArray = npt.NDArray[np.float32]

n_episodes = 4


#@parameterized_class(test_configs_fmt_class)
class D3RlPyDeterministicWrapperTest(unittest.TestCase):
    
    def setUp(self) -> None:
        self.algo_config = DQNConfig
    
    def test_call_no_scalers(self):
        
        #n_episodes = 4
        
        episode_length = 25
        action_size=2
        observation_shape=(4,)
        data_size = n_episodes * episode_length
        observations = create_observations(observation_shape, data_size)
        # if algo.get_action_type() == ActionSpace.CONTINUOUS:
        #     actions = np.random.random((data_size, action_size))
        # else:
        actions = np.random.randint(action_size, size=(data_size, 1))
        rewards: Float32NDArray = np.random.random(data_size).astype(np.float32)
        terminals: Float32NDArray = np.zeros(data_size, dtype=np.float32)
        for i in range(n_episodes):
            terminals[(i + 1) * episode_length - 1] = 1.0
        dataset = create_infinite_replay_buffer(
            EpisodeGenerator(observations, actions, rewards, terminals)()
        )
        algo=self.algo_config().create()
        
        init_trained_algo(
            algo=algo,
            dataset=dataset,
            )
        
        obs_tens = torch.tensor(observations)
        act_tenc = torch.tensor(actions)
        def _pref_mock(x:torch.Tensor):
            assert len(x.shape) == 2
            _slct = (obs_tens==x).all(axis=1)
            res = act_tenc[_slct,:]
            assert res.shape == (x.shape[0],1)
            assert _slct.numpy().sum() == x.shape[0]
            return res.squeeze()
        
        # Internally, the predict function 
        algo.impl.inner_predict_best_action = MagicMock(
            side_effect=_pref_mock
            )
        
        wrapper = D3RlPyDeterministicDiscreteWrapper(
            predict_func=algo.predict,
            action_dim=1
        )
        test_output = wrapper(x=torch.tensor(observations))
        rtol = actions.mean(keepdims=False)
        assert_allclose(
            actions,
            test_output.actions.numpy(),
            rtol = rtol
            )

    def test_call_obs_scaler(self):
        """Testing deterministic wrapper using and observation scaler. 
        Expected outcome: 
            - Observations are preprocessed by the predict function prior to 
            being fed to the algo.inner_predict_best_action function which has 
            been mocked
            - Therefore, _pref_mock uses obs_tens_trans to lookup the correct 
            action
            - However, no observation scalar is passed to the wrapper
        """
        
        episode_length = 25
        action_size=2
        observation_shape=(4,)
        data_size = n_episodes * episode_length
        observations = create_observations(observation_shape, data_size)
        # if algo.get_action_type() == ActionSpace.CONTINUOUS:
        #     actions = np.random.random((data_size, action_size))
        # else:
        actions = np.random.randint(action_size, size=(data_size, 1))
        rewards: Float32NDArray = np.random.random(data_size).astype(np.float32)
        terminals: Float32NDArray = np.zeros(data_size, dtype=np.float32)
        for i in range(n_episodes):
            terminals[(i + 1) * episode_length - 1] = 1.0
        dataset = create_infinite_replay_buffer(
            EpisodeGenerator(observations, actions, rewards, terminals)()
        )
        obs_scaler = StandardObservationScaler()
        obs_scaler.fit_with_transition_picker(
            episodes=dataset.episodes,
            transition_picker=dataset.transition_picker
        )
        algo= self.algo_config(
            observation_scaler=obs_scaler
            ).create()

        init_trained_algo(
            algo=algo,
            dataset=dataset,
            )
        
        obs_tens = torch.tensor(observations)
        act_tenc = torch.tensor(actions)
        obs_tens_trans = obs_scaler.transform(obs_tens)
        
        def _pref_mock(x:torch.Tensor):
            assert len(x.shape) == 2
            _slct = (obs_tens_trans==x).all(axis=1)
            res = act_tenc[_slct,:]
            assert res.shape == (x.shape[0],1)
            assert _slct.numpy().sum() == x.shape[0]
            return res.squeeze()
        
        algo.impl.inner_predict_best_action = MagicMock(
            side_effect=_pref_mock
            )
        
        wrapper = D3RlPyDeterministicDiscreteWrapper(
            predict_func=algo.predict,
            action_dim=1
        )
        
        test_output = wrapper(x=torch.tensor(observations))
        rtol = actions.mean(keepdims=False)
        assert_allclose(
            actions,
            test_output.actions.numpy(),
            rtol = rtol
            )


class D3RlPyStochasticWrapperTest(unittest.TestCase):

    def setUp(self) -> None:
        self.algo_config = SACConfig
    
    def test_call_no_scaler(self):
        
        episode_length = 25
        action_size=2
        observation_shape=(4,)
        data_size = n_episodes * episode_length
        observations = create_observations(observation_shape, data_size)
        actions = np.random.random((data_size, action_size))
        rewards: Float32NDArray = np.random.random(data_size).astype(np.float32)
        terminals: Float32NDArray = np.zeros(data_size, dtype=np.float32)
        for i in range(n_episodes):
            terminals[(i + 1) * episode_length - 1] = 1.0
        dataset = create_infinite_replay_buffer(
            EpisodeGenerator(observations, actions, rewards, terminals)()
        )
        algo=self.algo_config().create()
        
        init_trained_algo(
            algo=algo,
            dataset=dataset,
            )
        
        obs_tens = torch.tensor(observations)
        act_tens = torch.tensor(actions)
        mus = torch.tensor(np.random.random((data_size, action_size)))
        squashed_mus = torch.tanh(mus)
        std = torch.tensor(np.random.random((data_size, action_size)))
        
        def _pref_mock(x:torch.Tensor):
            assert len(x.shape) == 2
            _slct = (obs_tens==x).all(axis=1)
            assert _slct.numpy().sum() == x.shape[0]
            res_mus = mus[_slct,:]
            res_std = std[_slct,:]
            assert res_mus.shape == (x.shape[0],action_size)
            assert res_std.shape == (x.shape[0],action_size)
            squased_mu = squashed_mus[_slct,:]
            return ActionOutput(
                mu=res_mus,
                squashed_mu=squased_mu,
                logstd=res_std
            )
        
        algo.impl.policy.forward = MagicMock(
            side_effect=_pref_mock
            )
        
        wrapper = D3RlPyStochasticWrapper(
            policy_func=algo.impl.policy,
        )
        
        test_output = wrapper(
            state=obs_tens,
            action=act_tens
            )
        rtol = actions.mean(keepdims=False)
        dist = build_squashed_gaussian_distribution(
            action=ActionOutput(mu=mus,squashed_mu=squashed_mus,logstd=std)
        )
        with torch.no_grad():
            test_true = torch.exp(dist.log_prob(act_tens))
        assert_allclose(
            test_true,
            test_output.action_prs.numpy(),
            rtol = rtol
            )
        assert_allclose(
            actions,
            test_output.actions.numpy(),
            rtol = rtol
            )


    def test_call_action_scaler(self):
        """ If using an action scaler, the model expects to output a scaled
        action.
        - As such, the probabilities should be based on the scaled actions
        - However, the unscaled actions should be returned
        """        
        episode_length = 25
        action_size=2
        observation_shape=(4,)
        data_size = n_episodes * episode_length
        observations = create_observations(observation_shape, data_size)
        actions = np.random.random((data_size, action_size))
        rewards: Float32NDArray = np.random.random(data_size).astype(np.float32)
        terminals: Float32NDArray = np.zeros(data_size, dtype=np.float32)
        for i in range(n_episodes):
            terminals[(i + 1) * episode_length - 1] = 1.0
        dataset = create_infinite_replay_buffer(
            EpisodeGenerator(observations, actions, rewards, terminals)()
        )
        action_scaler = MinMaxActionScaler()
        action_scaler.fit_with_transition_picker(
            episodes=dataset.episodes,
            transition_picker=dataset.transition_picker
        )
        algo=self.algo_config(
            action_scaler=action_scaler
            ).create()
        
        init_trained_algo(
            algo=algo,
            dataset=dataset,
            )
        
        obs_tens = torch.tensor(observations)
        act_tens = torch.tensor(actions)
        act_tens_scaled = action_scaler.transform(act_tens)
        mus = torch.tensor(np.random.random((data_size, action_size)))
        squashed_mus = torch.tanh(mus)
        std = torch.tensor(np.random.random((data_size, action_size)))
        
        def _pref_mock(x:torch.Tensor):
            assert len(x.shape) == 2
            _slct = (obs_tens==x).all(axis=1)
            assert _slct.numpy().sum() == x.shape[0]
            res_mus = mus[_slct,:]
            res_std = std[_slct,:]
            assert res_mus.shape == (x.shape[0],action_size)
            assert res_std.shape == (x.shape[0],action_size)
            squased_mu = squashed_mus[_slct,:]
            return ActionOutput(
                mu=res_mus,
                squashed_mu=squased_mu,
                logstd=res_std
            )
        
        algo.impl.policy.forward = MagicMock(
            side_effect=_pref_mock
            )
        
        wrapper = D3RlPyStochasticWrapper(
            policy_func=algo.impl.policy,
            action_scaler=action_scaler
        )
                
        test_output = wrapper(
            state=obs_tens,
            action=act_tens
            )
        rtol = actions.mean(keepdims=False)
        dist = build_squashed_gaussian_distribution(
            action=ActionOutput(mu=mus,squashed_mu=squashed_mus,logstd=std)
        )
        with torch.no_grad():
            test_true = torch.exp(dist.log_prob(act_tens_scaled))
        assert_allclose(
            test_true,
            test_output.action_prs.numpy(),
            rtol = rtol
            )
        assert_allclose(
            actions,
            test_output.actions.numpy(),
            rtol = rtol
            )

    def test_call_obs_scaler(self):
        """Testing deterministic wrapper using and observation scaler. 
        Expected outcome: 
            - Observations are preprocessed by the wrapper prior to being fed 
            to the algo.impl.policy function which has been mocked
            - Therefore, _pref_mock uses obs_tens_trans to lookup
        """
        
        episode_length = 25
        action_size=2
        observation_shape=(4,)
        data_size = n_episodes * episode_length
        observations = create_observations(observation_shape, data_size)
        actions = np.random.random((data_size, action_size))

        rewards: Float32NDArray = np.random.random(data_size).astype(np.float32)
        terminals: Float32NDArray = np.zeros(data_size, dtype=np.float32)
        for i in range(n_episodes):
            terminals[(i + 1) * episode_length - 1] = 1.0
        dataset = create_infinite_replay_buffer(
            EpisodeGenerator(observations, actions, rewards, terminals)()
        )
        obs_scaler = StandardObservationScaler()
        obs_scaler.fit_with_transition_picker(
            episodes=dataset.episodes,
            transition_picker=dataset.transition_picker
        )
        algo=self.algo_config(
            observation_scaler=obs_scaler
            ).create()

        init_trained_algo(
            algo=algo,
            dataset=dataset,
            )

        obs_tens = torch.tensor(observations)
        obs_tens_scaled = obs_scaler.transform(obs_tens)
        act_tens = torch.tensor(actions)
        mus = torch.tensor(np.random.random((data_size, action_size)))
        squashed_mus = torch.tanh(mus)
        std = torch.tensor(np.random.random((data_size, action_size)))
        
        def _pref_mock(x:torch.Tensor):
            assert len(x.shape) == 2
            _slct = (obs_tens_scaled==x).all(axis=1)
            assert _slct.numpy().sum() == x.shape[0]
            res_mus = mus[_slct,:]
            res_std = std[_slct,:]
            assert res_mus.shape == (x.shape[0],action_size)
            assert res_std.shape == (x.shape[0],action_size)
            squased_mu = squashed_mus[_slct,:]
            return ActionOutput(
                mu=res_mus,
                squashed_mu=squased_mu,
                logstd=res_std
            )
        
        algo.impl.policy.forward = MagicMock(
            side_effect=_pref_mock
            )
        
        wrapper = D3RlPyStochasticWrapper(
            policy_func=algo.impl.policy,
            observation_scaler=obs_scaler
        )
        
        test_output = wrapper(
            state=obs_tens,
            action=act_tens
            )
        rtol = actions.mean(keepdims=False)
        dist = build_squashed_gaussian_distribution(
            action=ActionOutput(mu=mus,squashed_mu=squashed_mus,logstd=std)
        )
        with torch.no_grad():
            test_true = torch.exp(dist.log_prob(act_tens))
        assert_allclose(
            test_true,
            test_output.action_prs.numpy(),
            rtol = rtol
            )
        assert_allclose(
            actions,
            test_output.actions.numpy(),
            rtol = rtol
            )

    def test_call_obs_action_scaler(self):
        """ If using an action scaler, the model expects to output a scaled
        action.
        - As such, the probabilities should be based on the scaled actions
        - However, the unscaled actions should be returned
        Additionally, when using an observation scaler:
         - Observations are preprocessed by the wrapper prior to being fed 
         to the algo.impl.policy function which has been mocked
        - Therefore, _pref_mock uses obs_tens_trans to lookup
        """        
        episode_length = 25
        action_size=2
        observation_shape=(4,)
        data_size = n_episodes * episode_length
        observations = create_observations(observation_shape, data_size)
        actions = np.random.random((data_size, action_size))
        rewards: Float32NDArray = np.random.random(data_size).astype(np.float32)
        terminals: Float32NDArray = np.zeros(data_size, dtype=np.float32)
        for i in range(n_episodes):
            terminals[(i + 1) * episode_length - 1] = 1.0
        dataset = create_infinite_replay_buffer(
            EpisodeGenerator(observations, actions, rewards, terminals)()
        )
        action_scaler = MinMaxActionScaler()
        action_scaler.fit_with_transition_picker(
            episodes=dataset.episodes,
            transition_picker=dataset.transition_picker
        )
        obs_scaler = StandardObservationScaler()
        obs_scaler.fit_with_transition_picker(
            episodes=dataset.episodes,
            transition_picker=dataset.transition_picker
        )
        algo=self.algo_config(
            observation_scaler=obs_scaler,
            action_scaler=action_scaler
            ).create()
        
        init_trained_algo(
            algo=algo,
            dataset=dataset,
            )
        
        obs_tens = torch.tensor(observations)
        obs_tens_scaled = obs_scaler.transform(obs_tens)
        act_tens = torch.tensor(actions)
        act_tens_scaled = action_scaler.transform(act_tens)
        mus = torch.tensor(np.random.random((data_size, action_size)))
        squashed_mus = torch.tanh(mus)
        std = torch.tensor(np.random.random((data_size, action_size)))
        
        def _pref_mock(x:torch.Tensor):
            assert len(x.shape) == 2
            _slct = (obs_tens_scaled==x).all(axis=1)
            assert _slct.numpy().sum() == x.shape[0]
            res_mus = mus[_slct,:]
            res_std = std[_slct,:]
            assert res_mus.shape == (x.shape[0],action_size)
            assert res_std.shape == (x.shape[0],action_size)
            squased_mu = squashed_mus[_slct,:]
            return ActionOutput(
                mu=res_mus,
                squashed_mu=squased_mu,
                logstd=res_std
            )
        
        algo.impl.policy.forward = MagicMock(
            side_effect=_pref_mock
            )
        
        wrapper = D3RlPyStochasticWrapper(
            policy_func=algo.impl.policy,
            action_scaler=action_scaler,
            observation_scaler=obs_scaler
        )
                
        test_output = wrapper(
            state=obs_tens,
            action=act_tens
            )
        rtol = actions.mean(keepdims=False)
        dist = build_squashed_gaussian_distribution(
            action=ActionOutput(mu=mus,squashed_mu=squashed_mus,logstd=std)
        )
        with torch.no_grad():
            test_true = torch.exp(dist.log_prob(act_tens_scaled))
        assert_allclose(
            test_true,
            test_output.action_prs.numpy(),
            rtol = rtol
            )
        assert_allclose(
            actions,
            test_output.actions.numpy(),
            rtol = rtol
            )