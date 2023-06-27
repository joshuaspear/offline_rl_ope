from typing import List, cast
import logging
import numpy as np
import pandas as pd

from d3rlpy.metrics.scorer import _make_batches, WINDOW_SIZE, AlgoProtocol
from d3rlpy.dataset import Episode

from .utils import OPEEstimatorScorerBase, QueryCallbackBase

logger = logging.getLogger("offline_rl_ope")


class QueryScorer(OPEEstimatorScorerBase):
    
    def __init__(self, cache: QueryCallbackBase, query_key:str) -> None:
        super().__init__(cache)
        self.query_key = query_key
        
    def __call__(self, algo: AlgoProtocol, episodes: List[Episode]):
        return self.cache[self.query_key]


class DiscreteValueByActionCallback(QueryCallbackBase):
    
    def __init__(self, unique_action_vals:List, episodes: List[Episode]
                 ) -> None:
        super().__init__()
        self.unique_action_vals = unique_action_vals
        self.episodes = episodes
        
    
    def __call__(self, algo: AlgoProtocol, epoch, total_step):
        total_values = []
        total_actions = []
        for episode in self.episodes:
            for batch in _make_batches(episode, WINDOW_SIZE, algo.n_frames):
                values = algo.predict_value(batch.observations, batch.actions)
                total_actions += cast(np.ndarray, batch.actions).tolist()
                total_values += cast(np.ndarray, values).tolist()
        res = pd.DataFrame({"values": total_values, "actions":total_actions})
        res = res.groupby(by="actions", as_index=False)["values"].mean()
        res_dict = {key:val for key,val in zip(res["actions"], res["values"])}
        res_dict = {
            key: (res_dict[key] if key in res_dict.keys() else np.nan) 
            for key in self.unique_action_vals
            }
        self.cache = res_dict