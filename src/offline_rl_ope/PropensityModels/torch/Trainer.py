from abc import abstractmethod
import torch
import numpy as np
import pickle
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker

from ..base import PropensityTrainer
from ...types import (
    PropensityTorchBaseType, 
    TorchPolicyReturn,
    StateTensor, 
    ActionTensor
    )
from ...RuntimeChecks import check_array_dim

__all__ = [
    "TorchPropensityTrainer",
    "TorchClassTrainer",
    "TorchRegTrainer"
]

class TorchPropensityTrainer(PropensityTrainer):
    
    def __init__(
        self, 
        estimator:PropensityTorchBaseType, 
        gpu:bool,
        ) -> None:
        assert isinstance(estimator,PropensityTorchBaseType)
        assert isinstance(gpu,bool)
        self.estimator = estimator
        self.gpu = gpu
        if self.gpu:
            self.input_setup = self.load_to_gpu
        else:
            self.input_setup = self.load_pass
    
    def load_to_gpu(
        self, 
        x:torch.Tensor
        )->torch.Tensor:
        x = x.to(device='cuda')
        return x
    
    def load_pass(
        self, 
        x:torch.Tensor
        )->torch.Tensor:
        return x
    
    def to_cpu(self):
        self.input_setup = self.load_pass
        self.estimator = self.estimator.to(device='cpu')
    
    def to_gpu(self):
        self.input_setup = self.load_to_gpu
        self.estimator = self.estimator.to(device='cuda')
        
    def save(self, path:str) -> None:
        if self.gpu:
            self.to_cpu()
        with open(path, "wb") as f:
            pickle.dump(self, f)
        if self.gpu:
            self.to_gpu()
    
    @abstractmethod
    def predict(
        self, 
        x:StateTensor, 
        *args, 
        **kwargs
        ) -> ActionTensor:
        pass
    
    @abstractmethod
    def predict_proba(
        self, 
        x: StateTensor, 
        y: ActionTensor, 
        *args, 
        **kwargs
        ) -> Float[torch.Tensor, "traj_length 1"]:
        pass
    
    @jaxtyped(typechecker=typechecker)
    def policy_func(
        self, 
        x: StateTensor, 
        y: ActionTensor, 
        *args, 
        **kwargs
        ) -> TorchPolicyReturn:
        res = self.predict_proba(x=x,y=y)
        return TorchPolicyReturn(
            actions=y,
            action_prs=res
            )
    
        
               
class TorchClassTrainer(TorchPropensityTrainer):
    def __init__(
        self, 
        estimator:PropensityTorchBaseType, 
        gpu:bool
        ) -> None:
        assert isinstance(estimator,PropensityTorchBaseType)
        assert isinstance(gpu,bool)
        super().__init__(estimator=estimator, gpu=gpu)

    @jaxtyped(typechecker=typechecker)
    def predict(
        self, 
        x:StateTensor, 
        *args, 
        **kwargs
        ) -> ActionTensor:
        """Outputs the y values with highest likelihood given x.
        propense_res["out"] is expected to be of dimension: 
            (batch_size, n action values, n actions)

        Args:
            x (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        assert isinstance(x,torch.Tensor)
        x = self.input_setup(x)
        self.estimator.eval()
        propense_res = self.estimator(x)
        check_array_dim(propense_res["out"],3)
        # Take max over values
        res = torch.argmax(propense_res["out"], dim=1, keepdim=False).float()
        return res
    
    @jaxtyped(typechecker=typechecker)
    def predict_proba(
        self, 
        x: StateTensor, 
        y: ActionTensor, 
        *args, 
        **kwargs
        ) -> Float[torch.Tensor, "traj_length 1"]:
        """Outputs the normalised likelihood of each dimension of
        y given input x for classification.
        res["out"] is expected to be of dimension: 
            (batch_size, n action values, n actions)

        Args:
            x (torch.Tensor): _description_
            y (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        assert isinstance(x,torch.Tensor)
        assert isinstance(y,torch.Tensor)
        x = self.input_setup(x)
        self.estimator.eval()
        res = self.estimator(x)
        res_out = res["out"]
        check_array_dim(res_out,3)
        n_rows = res_out.shape[0]
        n_out = res_out.shape[2]
        dim_0_sub = np.arange(0,n_rows)[:,None]
        dim_1_sub = np.tile(np.arange(0,n_out), (n_rows,1))
        res_out = res_out[dim_0_sub,y.int(),dim_1_sub]
        assert res_out.shape == y.shape
        return res_out
        

class TorchRegTrainer(TorchPropensityTrainer):
    
    def __init__(
        self, 
        estimator:PropensityTorchBaseType, 
        dist_func:torch.distributions.Distribution, 
        gpu:bool
        ) -> None:
        assert isinstance(estimator,PropensityTorchBaseType)
        assert isinstance(gpu,bool)
        super().__init__(estimator=estimator, gpu=gpu)
        self.dist_func = dist_func
    
    @jaxtyped(typechecker=typechecker)
    def predict(
        self, 
        x:StateTensor, 
        *args, 
        **kwargs
        ) -> ActionTensor:
        x = self.input_setup(x)
        self.estimator.eval()
        propense_res = self.estimator(x)
        return propense_res["loc"]
    
    @jaxtyped(typechecker=typechecker)
    def predict_proba(
        self, 
        x: StateTensor, 
        y: ActionTensor, 
        *args, 
        **kwargs
        ) -> Float[torch.Tensor, "traj_length 1"]:
        assert isinstance(x,torch.Tensor)
        assert isinstance(y,torch.Tensor)
        x = self.input_setup(x)
        y = self.input_setup(y)
        self.estimator.eval()
        pred_res = self.estimator(x)
        d_f = self.dist_func(**pred_res)
        res = torch.exp(d_f.log_prob(y))
        return res