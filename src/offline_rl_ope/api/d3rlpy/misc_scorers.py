from typing import List, Sequence, Optional, cast
import logging
import numpy as np
import pandas as pd

from d3rlpy.metrics.evaluators import make_batches, WINDOW_SIZE
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.dataset import EpisodeBase, ReplayBuffer

from .utils import OPEEstimatorScorerBase, QueryCallbackBase

logger = logging.getLogger("offline_rl_ope")


class QueryScorer(OPEEstimatorScorerBase):
    
    def __init__(self, cache: QueryCallbackBase, query_key:str, 
                 episodes: Optional[Sequence[EpisodeBase]] = None
                 ) -> None:
        super().__init__(cache=cache, episodes=episodes)
        self.query_key = query_key
        
    def __call__(self, algo: QLearningAlgoProtocol, dataset: ReplayBuffer
                 ) -> float:
        return self.cache[self.query_key]


class DiscreteValueByActionCallback(QueryCallbackBase):
    """Callback class for calculating the on-policy average value estimates, 
    split by action.
    """
    
    def __init__(self, unique_action_vals:List, dataset: ReplayBuffer) -> None:
        super().__init__()
        self.unique_action_vals = unique_action_vals
        self.dataset = dataset
        
    
    def __call__(self, algo: QLearningAlgoProtocol, epoch:int, total_step:int):
        total_values = []
        total_actions = []
        for episode in self.dataset.episodes:
            for batch in make_batches(episode, WINDOW_SIZE, 
                                      self.dataset.transition_picker):
                values = algo.predict_value(batch.observations, batch.actions)
                total_actions += cast(
                    np.ndarray, batch.actions
                    ).reshape(-1).tolist()
                total_values += cast(np.ndarray, values).tolist()
        res = pd.DataFrame({
            "values": total_values, 
            "actions":total_actions
            })
        res = res.groupby(by="actions", as_index=False)["values"].mean()
        res_dict = {key:val for key,val in zip(res["actions"], res["values"])}
        res_dict = {
            key: (res_dict[key] if key in res_dict.keys() else np.nan) 
            for key in self.unique_action_vals
            }
        self.cache = res_dict