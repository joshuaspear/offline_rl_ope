from dataclasses import dataclass
import torch
from typing import Dict, Union, Protocol, runtime_checkable
import numpy as np
import pandas as pd
from jaxtyping import Float

__all__ = [
    "PropensityTorchOutputType",
    "StateTensor",
    "StateArray",
    "ActionTensor",
    "ActionArray",
    "RewardTensor",
    "WeightTensor",
    "SingleTrajSingleStepTensor",
    "PropensityTorchBaseType"
    ]

PropensityTorchOutputType = Dict[str,torch.Tensor]
ArrayType = Union[pd.DataFrame,np.ndarray,torch.Tensor]

state_dims = "traj_length n_state_features"
StateTensor = Float[torch.Tensor, state_dims]
StateArray = Float[np.ndarray, state_dims]

act_dims = "traj_length n_actions"
ActionTensor = Float[torch.Tensor, act_dims]
ActionArray = Float[np.ndarray, act_dims]


reward_dims = "traj_length 1"
RewardTensor = Float[torch.Tensor, reward_dims]

WeightTensor = Float[torch.Tensor, "n_trajectories max_length"]
SingleTrajSingleStepTensor = Float[torch.Tensor, "1"]

@runtime_checkable
class PropensityTorchBaseType(Protocol):
    
    def __call__(self, x:torch.Tensor) -> PropensityTorchOutputType:
        ...
    
    def eval(self)->None:
        ...


@runtime_checkable       
class PropensitySklearnContinuousType(Protocol):
    
    def predict_proba(self, X:StateArray) -> ActionArray:
        ...
    
    def predict(self, X:StateArray) -> ActionArray:
        ...

@dataclass
class TorchPolicyReturn:
    actions: Union[ActionTensor, None]
    action_prs: Union[Float[torch.Tensor, "traj_length 1"], None]
    

@dataclass
class NumpyPolicyReturn:
    actions: Union[ActionArray, None]
    action_prs: Union[Float[np.ndarray, "traj_length 1"], None]

    def get_torch_policy_return(self) -> TorchPolicyReturn:
        return TorchPolicyReturn(
            actions=torch.tensor(self.actions),
            action_prs=torch.tensor(self.action_prs)
        )