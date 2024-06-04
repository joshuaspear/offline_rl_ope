from abc import abstractmethod
from typing import Any, Callable, Dict, Tuple, List
from pymlrf.SupervisedLearning.torch import (
    train, validate_single_epoch
    )
from pymlrf.Structs.torch import DatasetOutput
from pymlrf.utils import set_seed
import numpy as np
from torch.utils.data import Dataset, DataLoader 
import torch
from torch.optim import Adam
import os

from offline_rl_ope.PropensityModels import PropensityTorchBase
from offline_rl_ope.types import PropensityTorchOutputType
from offline_rl_ope import logger


class GaussianLossWrapper:
    
    def __init__(self) -> None:
        self.scorer = torch.nn.GaussianNLLLoss()
    
    def __call__(
        self, 
        y_pred:PropensityTorchOutputType, 
        y_true:Dict[str,torch.Tensor]
        ) -> torch.Tensor:
        res = self.scorer(
            input=y_pred["loc"], 
            var=y_pred["scale"], 
            target=y_true["y"]
            )
        return res



class PropensityTrainingLoop:
    
    @abstractmethod
    def fit(self, *args, **kwargs) -> Dict[str,Any]:
        pass
    

class PropensityDataset(Dataset):
    
    def __init__(
        self, 
        x:np.array, 
        y:np.array
        ) -> None:
        super().__init__()
        if x.shape[0] != y.shape[0]:
            raise Exception
        if len(x.shape) != 2:
            raise Exception
        if len(y.shape) != 2:
            raise Exception
        self.x = x
        self.y = y
        self.__len = self.x.shape[0]
        
    def __len__(self)->int:
        return self.__len
    
    def __getitem__(self, idx:int)->Tuple[np.array]:
        return self.x[idx,:], self.y[idx,:]
        

class PropensityCollector:
    
    def __init__(self, trgt_type=torch.float) -> None:
        self.trgt_type=trgt_type
    
    def __call__(self, batch:List)->DatasetOutput:
        in_dict = {"x":[]}
        out_dict = {"y":[]}
        for row in batch:
            in_dict["x"].append(row[0])
            out_dict["y"].append(row[1]) 
        in_dict["x"] = torch.tensor(in_dict["x"], dtype=torch.float)
        out_dict["y"] = torch.tensor(out_dict["y"], dtype=self.trgt_type)
        return DatasetOutput(input=in_dict, output=out_dict)

propensity_collector = PropensityCollector()

class TorchTrainingLoop:
    
    def train(
        self, 
        model:PropensityTorchBase,
        x_train: np.array,
        y_train: np.array,
        x_val: np.array,
        y_val: np.array,
        batch_size:int,
        shuffle:bool,
        lr:float,
        gpu:bool,
        criterion:Callable,
        epochs:int,
        seed:int,
        save_dir:str,
        early_stopping_func:Callable        
        ):
        
        train_dataset = PropensityDataset(x=x_train, y=y_train)
        train_data_loader=DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=shuffle,
            collate_fn=propensity_collector
            )
        
        val_dataset = PropensityDataset(x=x_val, y=y_val)
        val_data_loader=DataLoader(
            dataset=val_dataset, batch_size=batch_size, shuffle=shuffle,
            collate_fn=propensity_collector
            )
        optimizer = Adam(
            params=model.parameters(),
            lr=lr
            )
        mo, optimal_epoch = train(
            model=model, 
            train_data_loader=train_data_loader, 
            val_data_loader=val_data_loader,
            gpu=gpu,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            logger=logger,
            seed=seed,
            save_dir=save_dir,
            early_stopping_func=early_stopping_func
            )
        metric_df = mo.all_metrics_to_df()
        metric_df.to_csv(os.path.join(save_dir, "training_metric_df.csv"))
        res = {}
        for key, metric in mo.metrics.items():
            res[key] = metric.value_dict[f"epoch_{optimal_epoch}"]
        res["optimal_epoch"] = optimal_epoch
        return res
    
    def test(
        self,
        model: PropensityTorchBase,
        x_test: np.array,
        y_test: np.array,
        gpu:bool,
        criterion:Callable,
        batch_size:int,
        seed:int
        ):
        dataset = PropensityDataset(x=x_test, y=y_test)
        data_loader=DataLoader(
            dataset=dataset, batch_size=batch_size, shuffle=False,
            collate_fn=propensity_collector
            )
        set_seed(seed)
        losses, preds = validate_single_epoch(
            model=model,
            data_loader=data_loader,
            gpu=gpu,
            criterion=criterion
        )
        res = {"mean_criterion_over_batch": np.mean(losses)}
        return res 
    
    def train_test(
        self,
        model:PropensityTorchBase,
        x_train: np.array,
        y_train: np.array,
        x_val: np.array,
        y_val: np.array,
        x_test: np.array,
        y_test: np.array,
        batch_size:int,
        shuffle:bool,
        lr:float,
        gpu:bool,
        criterion:Callable,
        epochs:int,
        seed:int,
        save_dir:str
        ):
        train_res = self.train(
            model=model, x_train=x_train, y_train=y_train, x_val=x_val, 
            y_val=y_val, batch_size=batch_size, shuffle=shuffle,
            lr=lr, gpu=gpu, criterion=criterion, epochs=epochs, seed=seed,
            save_dir=save_dir
        )
        test_res = self.test(
            model=model, x_test=x_test, y_test=y_test, gpu=gpu, 
            batch_size=batch_size, seed=seed
        )
        return {**train_res, **test_res}
        
    
torch_training_loop = TorchTrainingLoop()