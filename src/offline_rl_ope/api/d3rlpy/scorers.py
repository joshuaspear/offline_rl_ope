from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast
from typing_extensions import Protocol

import logging
import numpy as np
import pandas as pd

from d3rlpy.metrics.scorer import _make_batches, WINDOW_SIZE, AlgoProtocol
from d3rlpy.dataset import Episode

from ...components.utils import MultiOutputCache

logger = logging.getLogger("offline_rl_ope")

class DiscreteValueByActionCache(MultiOutputCache):
    
    def __init__(self, unique_action_vals:List) -> None:
        super().__init__(unique_values=unique_action_vals)
    
    def scoring_calc(self, algo: AlgoProtocol, episodes: List[Episode]):
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
        res_dict = {
            key: (res_dict[key] if key in res_dict.keys() else np.nan) 
            for key in self.unique_values
            }
        return res_dict


