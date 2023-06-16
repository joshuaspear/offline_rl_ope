from typing import Callable, Dict
import shutil
import os
import numpy as np
from d3rlpy.ope.fqe import _FQEBase
from d3rlpy.metrics.scorer import (AlgoProtocol)
from d3rlpy.dataset import MDPDataset 
from .utils import Wrapper

class FqeEvalD3rlpyWrap(Wrapper):
    """ Wrapper class for performing Fitted Q Evaluation
    """
    
    def __init__(self, scorers:Dict[str, Callable], fqe_cls:_FQEBase, 
                 model_init_kwargs:Dict, model_fit_kwargs:Dict,  
                 dataset:MDPDataset, env=None
                 ) -> None:
        super().__init__(scorers_nms=list(scorers.keys()))
        self.__scorers = scorers
        self.__dataset = dataset
        self.__fqe_cls = fqe_cls
        self.__model_init_kwargs = model_init_kwargs
        self.__model_fit_kwargs = model_fit_kwargs
        self.__logs_loc = os.path.join(os.getcwd(), "tmp_fqe_logs_loc")
        self.__cur_exp = 0
        os.mkdir(self.__logs_loc)
        self.__env = env
            
    def eval(self, algo: AlgoProtocol, epoch, total_step):
        fqe  = self.__fqe_cls(algo=algo, **self.__model_init_kwargs)
        if self.__env is not None:
            fqe.build_with_env(self.__env)
        else:
            fqe.build_with_dataset(self.__dataset)
        
        fqe.fit(self.__dataset.episodes, eval_episodes=self.__dataset.episodes, 
                scorers=self.__scorers, **self.__model_fit_kwargs, 
                logdir=self.__logs_loc, with_timestamp=False,
                experiment_name="EXP_{}".format(str(self.__cur_exp)))

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
        return res
                
    def clean_up(self):
        shutil.rmtree(self.__logs_loc)