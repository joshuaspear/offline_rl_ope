import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from typing import List
import pickle

from .base import PropensityTrainer
from ..types import Float32NDArray, NDArray

__all__ = [
    "MultiOutputMultiClassTrainer"
]

class MultiOutputMultiClassTrainer(PropensityTrainer):
    
    def __init__(
        self, 
        class_mins:List[float],
        class_maxs:List[float],
        estimator:MultiOutputClassifier, 
        epsilon_pr:float=0.0000001
        ) -> None:
        assert len(class_mins) == len(class_maxs)
        assert isinstance(estimator,MultiOutputClassifier)
        assert isinstance(epsilon_pr,float)
        if not isinstance(estimator, MultiOutputClassifier):
            raise Exception
        super().__init__()
        self.estimator = estimator
        self.epsilon_pr = epsilon_pr
        self.classes_def = []
        for mn,mx in zip(class_mins, class_maxs):
            self.classes_def.append(np.arange(mn, mx))
        self.__fitted_cls:List[NDArray] = []
        
        
    @staticmethod
    def __add_ms_cols(
        in_arr:Float32NDArray, 
        clas_def:NDArray, 
        fit_cls:NDArray, 
        in_val:float
        )->Float32NDArray:
        assert isinstance(in_arr,Float32NDArray)
        assert isinstance(clas_def,Float32NDArray)
        assert isinstance(fit_cls,Float32NDArray)
        assert isinstance(in_val,float)
        if len(fit_cls) == 1:
            # When there is only one class, the predict_proba outputs
            # P(X = 1) and P(X \neq 1)
            in_arr = in_arr[:,0][:,None]
        res = np.full((in_arr.shape[0], len(clas_def)), in_val)
        # Niavely remove the additional probability density from the other 
        # columns
        __addit_pr = (in_val*(len(clas_def)-len(fit_cls)))
        in_arr = in_arr - (__addit_pr/len(fit_cls))
        res[:,fit_cls.astype(int)] = in_arr
        return res
    
    def predict_proba(
        self, 
        x:Float32NDArray,
        y:Float32NDArray, 
        *args, 
        **kwargs
        ) -> Float32NDArray:
        assert isinstance(x,Float32NDArray)
        assert isinstance(y,Float32NDArray)
        assert x.shape[0] == y.shape[0]
        res = self.estimator.predict_proba(X=x, *args, **kwargs)
        probs = [
            self.__add_ms_cols(y,c_d,f_c, self.epsilon_pr) 
            for y,c_d,f_c in zip(res, self.classes_def, self.fitted_cls)
            ]
        
        num_output = y.shape[1]
        if num_output != len(probs):
            # Checks the number of action columns against the number predicted
            raise Exception(
                f"""Predicted outputs and true outputs differ
                num_output: {num_output}
                len(probs): {probs}
                """
                )
        res = []
        for i,out_prob in enumerate(probs):
            tmp_res = out_prob[
                np.arange(len(out_prob)),
                y[:,i].squeeze().astype(int)
                ]
            res.append(tmp_res.reshape(1,-1))
        res = np.concatenate(res, axis=0).prod(axis=0)
        return res
    
    def predict(
        self, 
        x:Float32NDArray, 
        *args, 
        **kwargs
        )->Float32NDArray:
        assert isinstance(x,Float32NDArray)
        return self.estimator.predict(X=x, *args, **kwargs)
    
    def save(self, path:str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @property
    def fitted_cls(self):
        msg = "Fitted class arrays are all empty"
        assert [len(i)>0 for i in self.__fitted_cls], msg
        return self.__fitted_cls
    
    @fitted_cls.setter
    def fitted_cls(self, val:List[NDArray]):
        self.__fitted_cls = val