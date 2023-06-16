import logging
from typing import Any, List

logger = logging.getLogger("offline_rl_ope")

class MultiOutputCache:
    """ Class to store a calculation result. Which can be quried a number of 
    times. The cache can be automatically refreshed after __value_call_max
    requests or manually refreshed.
    """
    def __init__(self, unique_values:List, manual:bool=False) -> None:
        if not isinstance(unique_values, List):
            raise TypeError("unique_values should be of type, List")
        self.unique_values = unique_values
        self.__value_call = 0
        self.__value_call_max = len(self.unique_values)
        self.__cache = {}
        self.__values_cached = False
        if manual:
            self.iterate_value_call = lambda x: True
        else:
            self.iterate_value_call = self.__auto_iterate
    
    @property
    def values_cached(self):
        return self.__values_cached
    
    def retrieve_item(self, value):
        self.iterate_value_call()
        return self.__cache[value]
    
    def flush(self):
        self.__value_call = 0
        self.cache = {}
        self.__values_cached = False

    def __auto_iterate(self):
        if self.__value_call < self.__value_call_max:
            self.__value_call += 1
            self.__values_cached = True
        else:
            self.flush()
    
    def scoring_calc(self, *args, **kwargs):
        raise NotImplementedError
    
    def run_store_score(self, *args, **kwargs):
        res_dict = self.scoring_calc(*args, **kwargs)
        if set(res_dict.keys()) != set(self.unique_values):
            logger.debug("res_dict: {}".format(res_dict))
            logger.debug("set(self.__unique_values): {}".format(
                set(self.unique_values)))
            logger.debug("set(res_dict.keys()): {}".format(
                set(res_dict.keys())))
            raise Exception("unique values and keys miss match")
        self.__cache = res_dict

class MultiOutputScorer:
    """Class to work with a MultiOutputCache object, representing a single 
    result. Running __call__ checks whether results are available in 
    MultiOutputCache and runs the calculation defined by the MultiOutputCache 
    if necessary. The result represented by this class is then retrieved using 
    self.__cache.retrieve_item
    """
    def __init__(self, value, cache:MultiOutputCache) -> None:
        self.__value = value
        self.__cache = cache
    
    def __call__(self, *args, **kwargs) -> Any:
        # Check to see whether the calculation has already been performed
        if not self.__cache.values_cached:
            self.__cache.run_store_score(*args, **kwargs)
        res = self.__cache.retrieve_item(self.__value)
        return res        