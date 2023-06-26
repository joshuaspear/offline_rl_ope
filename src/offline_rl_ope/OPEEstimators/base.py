from abc import ABCMeta, abstractmethod
import torch
from typing import List


class OPEEstimatorBase(metaclass=ABCMeta):
    
    @abstractmethod
    def predict(self, rewards:List[torch.Tensor], states:List[torch.Tensor], 
                actions:List[torch.Tensor], weights:torch.Tensor, 
                discount:float, is_msk:torch.Tensor
                )->float:
        pass
