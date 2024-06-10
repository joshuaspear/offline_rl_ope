from typing import Any, Callable, Dict, Union, Tuple, Sequence, Optional
import shutil
import os
import numpy as np
from d3rlpy.ope.fqe import FQEConfig, FQE, DiscreteFQE
from d3rlpy.dataset import MDPDataset
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.base import DeviceArg
from d3rlpy.logging import FileAdapterFactory
import gym
import gymnasium

from .base import QueryCallbackBase

Shape = Union[Sequence[int], Sequence[Sequence[int]]]
GymEnv = Union[gym.Env[Any, Any], gymnasium.Env[Any, Any]]
FQEImplInitArg = Union[GymEnv, Tuple[Shape, int]]

__all__ = [
    "FQECallback"
    ]


class FQECallback(QueryCallbackBase):
    """ Scorer class for performing Fitted Q Evaluation
    """
    
    def __init__(
        self, 
        scorers:Dict[str, Callable], 
        fqe_cls:Union[FQE, DiscreteFQE], 
        model_init_kwargs:Dict, 
        model_fit_kwargs:Dict, 
        dataset:MDPDataset, 
        fqe_impl_init:Optional[FQEImplInitArg]=None, 
        device:DeviceArg = False
        ) -> None:
        super().__init__(debug=False, debug_path="")
        self.__scorers = scorers
        self.__dataset = dataset
        self.__fqe_cls = fqe_cls
        self.__model_init_kwargs = model_init_kwargs
        self.__model_fit_kwargs = model_fit_kwargs
        self.__logs_loc = os.path.join(os.getcwd(), "tmp_fqe_logs_loc")
        self.__cur_exp = 0
        os.mkdir(self.__logs_loc)
        self.__fqe_impl_init = fqe_impl_init
        self.__device = device
        
    def debug_true(
        self,
        algo: QLearningAlgoProtocol, 
        epoch:int, 
        total_step:int
        ):
        pass
            
    def run(self, algo: QLearningAlgoProtocol, epoch:int, total_step:int):
        fqe_config = FQEConfig(**self.__model_init_kwargs)
        fqe = self.__fqe_cls(algo=algo, config=fqe_config, device=self.__device)
        if self.__fqe_impl_init is not None:
            if isinstance(self.__fqe_impl_init, Tuple):
                if len(self.__fqe_impl_init) != 2:
                    raise Exception
                fqe.create_impl(
                    observation_shape=self.__fqe_impl_init[0],
                    action_size=self.__fqe_impl_init[1], 
                    )
            else:    
                fqe.build_with_env(self.__fqe_impl_init)
        
        _msg = "Must provide n_steps for FQE training"
        assert "n_steps" in self.__model_fit_kwargs, _msg
        _msg = "Must provide n_steps_per_epoch for FQE training"
        assert "n_steps_per_epoch" in self.__model_fit_kwargs, _msg
        
        res = fqe.fit(
            self.__dataset, 
            evaluators=self.__scorers, 
            **self.__model_fit_kwargs, 
            logger_adapter=FileAdapterFactory(root_dir=self.__logs_loc),
            with_timestamp=False, 
            experiment_name=f"EXP_{str(self.__cur_exp)}"
            )

        res = res[-1][1]
        self.__cur_exp += 1
        self.cache = res

    
    def clean_up(self):
        shutil.rmtree(self.__logs_loc)