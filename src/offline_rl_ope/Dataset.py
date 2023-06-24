import dataclasses
import torch
from typing import List

@dataclasses.dataclass(init=True)
class ISEpisode:
    
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor