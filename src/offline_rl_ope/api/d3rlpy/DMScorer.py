from typing import Callable, Dict, Union
import shutil
import os
import numpy as np
from d3rlpy.ope.fqe import FQEConfig, FQE, DiscreteFQE
from d3rlpy.dataset import MDPDataset
from d3rlpy.interface import QLearningAlgoProtocol
from d3rlpy.base import DeviceArg
from d3rlpy.logging import FileAdapterFactory

from .utils import QueryCallbackBase

class FQECallback(QueryCallbackBase):
    """ Scorer class for performing Fitted Q Evaluation
    """
    
    def __init__(self, scorers:Dict[str, Callable], 
                 fqe_cls:Union[FQE, DiscreteFQE], model_init_kwargs:Dict, 
                 model_fit_kwargs:Dict, dataset:MDPDataset, env=None, 
                 device:DeviceArg = False
                 ) -> None:
        self.__scorers = scorers
        self.__dataset = dataset
        self.__fqe_cls = fqe_cls
        self.__model_init_kwargs = model_init_kwargs
        self.__model_fit_kwargs = model_fit_kwargs
        self.__logs_loc = os.path.join(os.getcwd(), "tmp_fqe_logs_loc")
        self.__cur_exp = 0
        os.mkdir(self.__logs_loc)
        self.__env = env
        self.__device = device
            
    def __call__(self, algo: QLearningAlgoProtocol, epoch:int, total_step:int):
        fqe_config = FQEConfig(**self.__model_init_kwargs)
        fqe = self.__fqe_cls(algo=algo, config=fqe_config, device=self.__device)
        if self.__env is not None:
            fqe.build_with_env(self.__env)
        else:
            fqe.build_with_dataset(self.__dataset)
        
        fqe.fit(self.__dataset,  evaluators=self.__scorers, 
                **self.__model_fit_kwargs, 
                logger_adapter=FileAdapterFactory(root_dir=self.__logs_loc),
                with_timestamp=False, 
                experiment_name=f"EXP_{str(self.__cur_exp)}")

        res:Dict = {}
        for scr in self.__scorers:
            __file_path = os.path.join(
                self.__logs_loc, "EXP_{}".format(self.__cur_exp), 
                "{}.csv".format(scr))
            lines:np.array = np.genfromtxt(__file_path, delimiter=',')
            if len(lines.shape) == 1:
                lines = lines.reshape(-1,1)
            res[scr] = lines[-1:,-1].item()
        self.__cur_exp += 1
        self.cache = res

    
    def clean_up(self):
        shutil.rmtree(self.__logs_loc)