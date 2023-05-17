from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast
from typing_extensions import Protocol

import numpy as np
import pandas as pd

from d3rlpy.metrics.scorer import _make_batches, WINDOW_SIZE, AlgoProtocol
from d3rlpy.dataset import Episode

def discrete_value_by_action(
    algo: AlgoProtocol, episodes: List[Episode]
    ) -> float:
    r"""
    """
    total_values = []
    total_actions = []
    for episode in episodes:
        for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
            values = algo.predict_value(batch.observations, batch.actions)
            total_actions += cast(np.ndarray, batch.actions).tolist()
            total_values += cast(np.ndarray, values).tolist()
    res = pd.DataFrame({"values": total_values, "actions":total_actions})
    res = res.groupby(by="actions")["values"].mean()
    return res