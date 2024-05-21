import torch
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker

from .Discrete import MultiOutputMultiClassTrainer
from ...types import StateTensor, ActionTensor

class SklearnTorchTrainerWrapper:
    
    def __init__(
        self, 
        sklearn_trainer:MultiOutputMultiClassTrainer
        ) -> None:
        assert isinstance(sklearn_trainer,MultiOutputMultiClassTrainer)
        self.__sklearn_trainer = sklearn_trainer

    @jaxtyped(typechecker=typechecker)
    def __call__(
        self, 
        y:ActionTensor, 
        x:StateTensor
        )->torch.Tensor:
        # assert isinstance(y,torch.Tensor)
        # assert isinstance(x,torch.Tensor)
        res = torch.Tensor(
            self.__sklearn_trainer.predict_proba(
                y=y.detach().cpu().numpy(),
                x=x.detach().cpu().numpy()
                )
            )
        return res