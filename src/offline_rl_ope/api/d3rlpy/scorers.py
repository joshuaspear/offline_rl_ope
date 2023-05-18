from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast
from typing_extensions import Protocol

import numpy as np
import pandas as pd

from d3rlpy.metrics.scorer import _make_batches, WINDOW_SIZE, AlgoProtocol
from d3rlpy.dataset import Episode

class discrete_value_by_action_scorer:
    
    def __init__(self, unique_action_vals) -> None:
        self.unique_action_vals = unique_action_vals

    def __call__(
        self, algo: AlgoProtocol, episodes: List[Episode]
        ) -> np.array:
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
        res = res.groupby(by="actions", as_index=False)["values"].mean()
        res_dict = {key:val for key,val in zip(res["actions"], res["values"])}
        for act in self.unique_action_vals:
            if act not in res_dict.keys():
                res_dict[act] = np.nan
        res = np.array(list(map(
            lambda x: res_dict[x], self.unique_action_vals
            )))
        return res