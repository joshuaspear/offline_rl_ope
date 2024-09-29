from d3rlpy.algos import QLearningAlgoBase, QLearningAlgoImplBase
from d3rlpy.base import LearnableConfig
from d3rlpy.logging import NoopAdapterFactory
from d3rlpy.dataset import ReplayBuffer
import numpy.typing as npt
from typing import Sequence, Union, Any, overload, cast
import torch
from unittest.mock import Mock
import numpy as np


NDArray = npt.NDArray[Any]
Float32NDArray = npt.NDArray[np.float32]
Int32NDArray = npt.NDArray[np.int32]
UInt8NDArray = npt.NDArray[np.uint8]
DType = npt.DTypeLike

Observation = Union[NDArray, Sequence[NDArray]]
ObservationSequence = Union[NDArray, Sequence[NDArray]]
Shape = Union[Sequence[int], Sequence[Sequence[int]]]
TorchObservation = Union[torch.Tensor, Sequence[torch.Tensor]]



@overload
def create_observations(
    observation_shape: Sequence[int], length: int, dtype: DType = np.float32,
    seed:int = 1
) -> NDArray: ...


@overload
def create_observations(
    observation_shape: Sequence[Sequence[int]],
    length: int,
    dtype: DType = np.float32,
    seed:int = 1
) -> Sequence[NDArray]: ...


def create_observations(
    observation_shape: Shape, length: int, dtype: DType = np.float32, 
    seed:int = 1
) -> ObservationSequence:
    observations: ObservationSequence
    np.random.seed(seed=seed)
    if isinstance(observation_shape[0], (list, tuple)):
        observations = [
            np.random.random((length, *shape)).astype(dtype)
            for shape in cast(Sequence[Sequence[int]], observation_shape)
        ]
    else:
        observations = np.random.random(
            (length, *observation_shape)
            ).astype(
            dtype
        )
    return observations

def init_trained_algo(
    algo: QLearningAlgoBase[QLearningAlgoImplBase, LearnableConfig],
    dataset:ReplayBuffer
    ):
    algo.update = Mock(return_value={"loss": np.random.random()})  # type: ignore
    n_batch = algo.config.batch_size
    n_steps = 10
    n_steps_per_epoch = 5
    n_epochs = n_steps // n_steps_per_epoch
    # data_size = n_episodes * episode_length

    # check fit
    results = algo.fit(
        dataset,
        n_steps=n_steps,
        n_steps_per_epoch=n_steps_per_epoch,
        logger_adapter=NoopAdapterFactory(),
        show_progress=False,
    )