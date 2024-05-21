import numpy as np
from sklearn.multioutput import MultiOutputClassifier
from typing import List, Union
import pickle
from jaxtyping import jaxtyped, Float, Int
from typeguard import typechecked as typechecker

from ...types import StateArray, ActionArray
from ..base import PropensityTrainer

__all__ = [
    "MultiOutputMultiClassTrainer"
]

class MultiOutputMultiClassTrainer(PropensityTrainer):
    
    def __init__(
        self, 
        theoretical_action_classes:List[np.ndarray],
        estimator:MultiOutputClassifier, 
        epsilon_pr:float=0.0000001
        ) -> None:
        """
        Args:
            theoretical_action_classes (List[np.ndarray]): List of numpy arrays
            defining the theoretical classes for each action
            estimator (MultiOutputClassifier): _description_
            epsilon_pr (float, optional): _description_. Defaults to 0.0000001.
        """
        assert isinstance(estimator,MultiOutputClassifier)
        assert isinstance(epsilon_pr,float)
        super().__init__()
        self.estimator = estimator
        self.epsilon_pr = epsilon_pr
        self.classes_def = theoretical_action_classes
        self.__fitted_cls:List[np.ndarray] = []
        
    @jaxtyped(typechecker=typechecker)
    @staticmethod
    def __add_ms_cols(
        in_arr:Union[
            Float[np.ndarray, "traj_length action_options"],
            Float[np.ndarray, "traj_length 2"]
            ], 
        clas_def:Int[np.ndarray, "theoretical_action_options"], 
        fit_cls:Int[np.ndarray, "action_options"], 
        in_val:float
        )->Float[np.ndarray, "traj_length theoretical_action_options"]:
        """Method to impute probabiltiies for actions not included in the 
        training set i.e., assume: 
            - The theoretical action space is [0,1,2]
            - The training data only contained [0,1]
        Then a probability for action 2 should be imputed. 

        Args:
            in_arr (np.ndarray): Array of dimension 
            (traj_length, action_options) defining the probability of each class
            according to the model i.e. in_arr.shape[1] <= len(clas_def) 
            however, in_arr.shape[1] == len(fit_cls)
            clas_def (np.ndarray): Theoretical number of classes for action
            fit_cls (np.ndarray): Actual fitted classes for the action
            in_val (float): Probability to impute for missing actions

        Returns:
            np.ndarray: Array defining the state condition probability of each 
            action option
        """
        # assert isinstance(in_arr,np.ndarray)
        # assert isinstance(clas_def,np.ndarray)
        # assert isinstance(fit_cls,np.ndarray)
        # assert isinstance(in_val,float)
        assert len(fit_cls) == in_arr.shape[1]
        
        if len(fit_cls) == 2:
            # When there is only one class i.e., 0 and 1, the predict_proba 
            # outputs P(X = 1) and P(X \neq 1)
            # Therefore, take the first column
            in_arr = in_arr[:,0][:,None]
        res = np.full((in_arr.shape[0], len(clas_def)), in_val)
        # Niavely remove the additional probability density from the other 
        # columns
        __addit_pr = (in_val*(len(clas_def)-len(fit_cls)))
        in_arr = in_arr - (__addit_pr/len(fit_cls))
        res[:,fit_cls.astype(int)] = in_arr
        return res
    
    @jaxtyped(typechecker=typechecker)
    def predict_proba(
        self, 
        x:StateArray,
        y:ActionArray, 
        *args, 
        **kwargs
        )->Float[np.ndarray, "traj_length 1"]:
        """Calculates the joint probability of the given actions conditional 
        on the given state according to the policy in estimator
        
        Args:
            x (np.ndarray): Array of dimension (traj_length, n_state_features)
            defining the state
            y (np.ndarray): Array of dimension (traj_length, n_actions)
            defining the actions
            
        Returns:
            np.ndarray: Array of dimension (traj_length,1) defining the joint 
            probability of the actions conditional on the state.
        """
        # assert isinstance(x,np.ndarray)
        # assert isinstance(y,np.ndarray)
        # assert x.shape[0] == y.shape[0]
        res = self.estimator.predict_proba(X=x, *args, **kwargs)
        # Res is out output List[np.array] where the length of the list is 
        # equivalent to n actions and the dim of each np.array is equivalent to 
        # the number of options per action 
        assert len(res) == y.shape[1]
        probs = [
            self.__add_ms_cols(y,c_d,f_c, self.epsilon_pr) 
            for y,c_d,f_c in zip(res, self.classes_def, self.fitted_cls)
            ]
        res = []
        for i,out_prob in enumerate(probs):
            tmp_res = out_prob[
                np.arange(len(out_prob)),
                y[:,i].squeeze().astype(int)
                ]
            res.append(tmp_res.reshape(-1,1))
        res = np.concatenate(res, axis=1).prod(axis=1,keepdims=True)
        return res
    
    @jaxtyped(typechecker=typechecker)
    def predict(
        self, 
        x:StateArray, 
        *args, 
        **kwargs
        )->ActionArray:
        """Predicts the actions given the provided state according to the policy
        defined in estimator.

        Args:
            x (np.ndarray): Array of dimension (traj_length, n_state_features)
            defining the state

        Returns:
            np.ndarray: Array of dimension (traj_length,n_actions) defining the 
            predicted action values given the state.
        """
        # assert isinstance(x,np.ndarray)
        return self.estimator.predict(X=x, *args, **kwargs)
    
    def save(self, path:str) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)
    
    @property
    def fitted_cls(self)->List[np.ndarray]:
        msg = "Fitted class arrays are all empty"
        assert [len(i)>0 for i in self.__fitted_cls], msg
        return self.__fitted_cls
    
    @fitted_cls.setter
    def fitted_cls(
        self, 
        val:List[np.ndarray]
        ):
        self.__fitted_cls = val