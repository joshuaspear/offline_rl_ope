from unittest import TestCase
import torch
import torch.nn as nn
import numpy as np
import numpy.testing as nt

from offline_rl_ope.PropensityModels.torch import (
    TorchPropensityTrainer, TorchClassTrainer, TorchRegTrainer)


in_y = torch.tensor(np.array([0,0,1,2,0,0,1,0]).reshape(-1,1))

in_x = torch.tensor(
    np.array(
        [[1,2],[3,4],[5,6],[7,8],[1,2],[3,4],[5,6],[7,8]]
    )
)
res = {
    "out": torch.tensor(
        [
            [[0.1,0.8,0.1]],
            [[0.6,0.2,0.2]],
            [[0.3,0.5,0.2]],
            [[0.2,0.4,0.4]],
            [[0.1,0.8,0.1]],
            [[0.6,0.2,0.2]],
            [[0.3,0.5,0.2]],
            [[0.2,0.4,0.4]]
        ]
    ).view(8,3,1)
}

res_predict_true = res["out"].argmax(dim=1, keepdim=False)
res_predict_proba = []
for i in zip(res["out"],in_y):
    res_predict_proba.append(i[0][i[1].item()])
res_predict_proba = torch.concat(res_predict_proba)[:,None]

in_y2D = torch.tensor(np.array(
    [[0,0],[0,2],[1,1],[2,2],[0,2],[0,1],[1,2],[0,1]]
    ))
res2D = {
    "out": torch.tensor(
        [
            [[0.1,0.8,0.1],[0.6,0.2,0.2]],
            [[0.6,0.2,0.2],[0.6,0.2,0.2]],
            [[0.3,0.5,0.2],[0.1,0.8,0.1]],
            [[0.2,0.4,0.4],[0.1,0.8,0.1]],
            [[0.1,0.8,0.1],[0.2,0.4,0.4]],
            [[0.6,0.2,0.2],[0.3,0.5,0.2]],
            [[0.3,0.5,0.2],[0.2,0.4,0.4]],
            [[0.2,0.4,0.4],[0.1,0.8,0.1]]
        ]
    ).view(8,3,2)
}

class PropensityTorchMock:
    
    def __init__(self, d) -> None:
        if d == 1:
            self.test_out = res
        elif d== 2:
            self.test_out = res2D
    
    def eval(self):
        pass
    
    def __call__(self, x):
        return self.test_out


estimator_mock1D = PropensityTorchMock(d=1)
estimator_mock2D = PropensityTorchMock(d=2)

class TorchClassTrainerTest(TestCase):
        
    def test_predict1D(self) -> torch.Tensor:
        trainer = TorchClassTrainer(
            estimator=estimator_mock1D,gpu=False
            )
        pred = trainer.predict(x=in_x.float())
        assert len(pred.shape) == 2
        assert isinstance(pred, torch.Tensor)
        nt.assert_array_equal(
            pred.numpy(), res_predict_true.numpy()
            )
        
        
    
    def test_predict_proba1D(self) -> torch.Tensor:
        trainer = TorchClassTrainer(
            estimator=estimator_mock1D,gpu=False
            )
        pred = trainer.predict_proba(x=in_x.float(),y=in_y.float())
        assert len(pred.shape) == 2
        assert isinstance(pred, torch.Tensor)
        nt.assert_array_equal(
            pred.numpy(), res_predict_proba.numpy()
            )
    
    # def test_predict2D(
    #     self, 
    #     x:torch.Tensor, 
    #     *args, 
    #     **kwargs
    #     ) -> torch.Tensor:
    #     """Outputs the y values with highest likelihood given x

    #     Args:
    #         x (torch.Tensor): _description_

    #     Returns:
    #         torch.Tensor: _description_
    #     """
    #     x = self.input_setup(x)
    #     self.estimator.eval()
    #     res = self.estimator(x)
    #     # Take max over values
    #     res = torch.argmax(res["out"], dim=1, keepdim=False)
    #     return res
    
    # def test_predict_proba2D(
    #     self, 
    #     x: torch.Tensor, 
    #     y: torch.Tensor, 
    #     *args, 
    #     **kwargs
    #     ) -> torch.Tensor:
    #     """Outputs the normalised likelihood of each dimension of
    #     y given input x for classification.

    #     Args:
    #         x (torch.Tensor): _description_
    #         y (torch.Tensor): _description_

    #     Returns:
    #         torch.Tensor: _description_
    #     """
    #     x = self.input_setup(x)
    #     self.estimator.eval()
    #     res = self.estimator(x)
    #     res_out = res["out"].cpu().detach()
    #     n_rows = res_out.shape[0]
    #     n_out = res_out.shape[2]
    #     dim_0_sub = np.arange(0,n_rows)[:,None]
    #     dim_1_sub = np.tile(np.arange(0,n_out), (n_rows,1))
    #     res_out = res_out[dim_0_sub,y.int(),dim_1_sub]
    #     return res_out