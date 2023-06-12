from typing import Any, Callable, Iterator, List, Optional, Tuple, Union, cast
from typing_extensions import Protocol

import logging
import numpy as np
import pandas as pd

from d3rlpy.metrics.scorer import _make_batches, WINDOW_SIZE, AlgoProtocol
from d3rlpy.dataset import Episode

logger = logging.getLogger("offline_rl_ope")

class MultiOutputCache:
    
    def __init__(self, unique_values:List) -> None:
        if not isinstance(unique_values, List):
            raise TypeError("unique_values should be of type, List")
        self.unique_values = unique_values
        self.__value_call = 0
        self.__value_call_max = len(self.unique_values)
        self.__cache = {}
        self.__values_cached = False
    
    @property
    def values_cached(self):
        return self.__values_cached
    
    def retrieve_item(self, value):
        self.__iterate_value_call()
        return self.__cache[value]

    def __iterate_value_call(self):
        if self.__value_call < self.__value_call_max:
            self.__value_call += 1
            self.__values_cached = True
        else:
            self.__value_call = 0
            self.cache = {}
            self.__values_cached = False
    
    def scoring_calc(self, algo: AlgoProtocol, episodes: List[Episode]):
        raise NotImplementedError
    
    def run_store_score(self, algo: AlgoProtocol, episodes: List[Episode]):
        res_dict = self.scoring_calc(algo=algo, episodes=episodes)
        if set(res_dict.keys()) != set(self.unique_values):
            logger.debug("res_dict: {}".format(res_dict))
            logger.debug("set(self.__unique_values): {}".format(
                set(self.unique_values)))
            logger.debug("set(res_dict.keys()): {}".format(
                set(res_dict.keys())))
            raise Exception("unique values and keys miss match")
        self.__cache = res_dict



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


class MultiOutputScorer:
    
    def __init__(self, value, cache:MultiOutputCache) -> None:
        self.__value = value
        self.__cache = cache
    
    def __call__(
        self, algo: AlgoProtocol, episodes: List[Episode]
        ) -> np.array:
        
        # Check to see whether the calculation has already been performed
        if not self.__cache.values_cached:
            self.__cache.run_store_score(algo=algo, episodes=episodes)
        res = self.__cache.retrieve_item(self.__value)
        return res        