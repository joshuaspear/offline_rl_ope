import unittest
from unittest.mock import MagicMock
import torch
import numpy as np
from d3rlpy.algos import SACConfig
from d3rlpy.dataset import (
    EpisodeGenerator,
    create_infinite_replay_buffer,
)
from d3rlpy.preprocessing import StandardRewardScaler
from offline_rl_ope.OPEEstimators.DirectMethod import D3rlpyQlearnDM
import numpy.typing as npt
from ..d3rlpy_test_utils import init_trained_algo, create_observations

Float32NDArray = npt.NDArray[np.float32]

n_episodes = 4

gamma = 0.99

#dm_model = MagicMock(spec=DirectMethodBase)

class AlgoMock:
    
    def predict():
        pass
    
    def predict_value():
        pass

class D3rlpyQlearnDMTest(unittest.TestCase):
    
    def setUp(self) -> None:
        episode_length = 25
        action_size=2
        observation_shape=(4,)
        data_size = n_episodes * episode_length
        observations = create_observations(observation_shape, data_size, seed=1)
        np.random.seed(1)
        actions = np.random.random((data_size, action_size))
        np.random.seed(1)
        rewards: Float32NDArray = np.random.random(data_size).astype(np.float32)
        terminals: Float32NDArray = np.zeros(data_size, dtype=np.float32)
        for i in range(n_episodes):
            terminals[(i + 1) * episode_length - 1] = 1.0
        dataset = create_infinite_replay_buffer(
            EpisodeGenerator(observations, actions, rewards, terminals)()
        )
        self.algo=SACConfig().create()
        init_trained_algo(
            algo=self.algo,
            dataset=dataset,
            )
        
        self.obs = observations
        self.actions = actions
        self.obs_tens = torch.tensor(observations)
        self.actions_tens = torch.tensor(actions)
        self.rewards = rewards
        
        def _predict_func(x:np.array)->np.array:
            assert len(x.shape) == 2
            _slct = (observations==x).all(axis=1)
            assert _slct.sum() == x.shape[0]
            return actions[_slct,:].squeeze()
        
        self.algo.predict = MagicMock(
            side_effect=_predict_func
            )
        
        def _predict_value_func(x:np.array,action:np.array)->np.array:
            assert len(x.shape) == 2
            _slct = np.concatenate([
                (observations==x).all(axis=1, keepdims=True),
                (actions==action).all(axis=1, keepdims=True),
                ], axis=1).all(axis=1)
            assert _slct.sum() == x.shape[0]
            return rewards[_slct]
        
        self.algo.predict_value = MagicMock(
            side_effect=_predict_value_func
        )
        
        reward_scaler = StandardRewardScaler()
        self.algo_r_scale = SACConfig(reward_scaler=reward_scaler).create()
        init_trained_algo(
            algo=self.algo_r_scale,
            dataset=dataset,
            )
        
        self.rewards_scaled = self.algo_r_scale._config.reward_scaler.transform_numpy(rewards)
        
        def _predict_value_func_r_scaled(x:np.array,action:np.array)->np.array:
            assert len(x.shape) == 2
            _slct = np.concatenate([
                (observations==x).all(axis=1, keepdims=True),
                (actions==action).all(axis=1, keepdims=True),
                ], axis=1).all(axis=1)
            assert _slct.sum() == x.shape[0]
            return self.rewards_scaled[_slct]
        
        self.algo_r_scale.predict = MagicMock(
            side_effect=_predict_func
            )
        
        self.algo_r_scale.predict_value = MagicMock(
            side_effect=_predict_value_func_r_scaled
        )

    def test_calculate_q_no_scalers(self):
        direct_method = D3rlpyQlearnDM(model=self.algo)
        pred = direct_method.calculate_q(
            state=self.obs_tens,
            action=self.actions_tens
            )
        np.testing.assert_allclose(
            pred.squeeze(),
            self.rewards,
            rtol=0.001
        )
    
    def test_calculate_v_no_scalers(self):
        direct_method = D3rlpyQlearnDM(model=self.algo)
        pred = direct_method.calculate_v(
            state=self.obs_tens
            )
        np.testing.assert_allclose(
            pred.squeeze(),
            self.rewards,
            rtol=0.001
        )
        
    def test_calculate_q_rew_scalers(self):
        direct_method = D3rlpyQlearnDM(model=self.algo_r_scale)
        pred = direct_method.calculate_q(
            state=self.obs_tens,
            action=self.actions_tens
            )
        np.testing.assert_allclose(
            pred.squeeze(),
            self.rewards,
            rtol=0.001
        )
    
    def test_calculate_v_rew_scalers(self):
        direct_method = D3rlpyQlearnDM(model=self.algo_r_scale)
        pred = direct_method.calculate_v(
            state=self.obs_tens
            )
        print(pred.squeeze()-self.rewards)
        np.testing.assert_allclose(
            pred.squeeze(),
            self.rewards,
            rtol=0.001
        )