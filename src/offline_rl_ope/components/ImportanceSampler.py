from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple
import torch
from torch.nn.functional import pad
from jaxtyping import jaxtyped, Float
from typeguard import typechecked as typechecker


from ..RuntimeChecks import check_array_dim, check_array_shape
from .Policy import BasePolicy
from .. import logger
from ..types import (StateTensor,ActionTensor,WeightTensor)


__all__ = [
    "ISWeightCalculator", "ImportanceSampler", "VanillaIS", "PerDecisionIS",
    "ISWeightOrchestrator"
    ]


class ISWeightCalculator:
    def __init__(self, behav_policy:BasePolicy) -> None:
        assert isinstance(behav_policy,BasePolicy)
        self.__behav_policy = behav_policy
        self.is_weights = torch.empty(0)
        self.weight_msk = torch.empty(0)
        self.policy_actions:List[torch.Tensor] = [torch.empty(0)]
    
    @jaxtyped(typechecker=typechecker)
    def get_traj_w(
        self, 
        states:StateTensor, 
        actions:ActionTensor, 
        eval_policy:BasePolicy
        )->Float[torch.Tensor, "traj_length"]:
        """Function to calculate the timestep IS weights over a trajectory i.e., 
        for each timestep (t) Tensor(\pi_{e}(a_{t}|s_{t})/\pi_{b}(a_{t}|s_{t}))
        Args:
            states (torch.Tensor): Tensor of dimension (traj_length, state size)
            actions (torch.Tensor): Tensor of dimension 
                (traj_length, action size)

        Returns:
            torch.Tensor: Tensor of dimension (traj_length) defining the 
            propensity weights for the input trajectory
        """
        # if (len(states.shape) != 2) | (len(actions.shape) != 2):
        #     logger.debug("states.shape: {}".format(states.shape))
        #     logger.debug("actions.shape: {}".format(actions.shape))
        #     raise Exception("State and actions should have 2 dimensions")
        # check_array_dim(states,2)
        # check_array_dim(actions,2)
        # assert isinstance(states, torch.Tensor)
        # assert isinstance(actions, torch.Tensor)
        #assert isinstance(eval_policy, BasePolicy)
        
        with torch.no_grad():
            behav_probs = self.__behav_policy(
                action=actions, state=states
                )
            check_array_dim(behav_probs,2)
            assert behav_probs.shape[1] == 1
            #logger.debug("behav_probs: {}".format(behav_probs))
            eval_probs = eval_policy(action=actions, state=states)
            check_array_dim(eval_probs,2)
            assert eval_probs.shape[1] == 1
            assert behav_probs.shape == eval_probs.shape 
        #logger.debug("eval_probs: {}".format(eval_probs))
        weight_array = eval_probs/behav_probs
        weight_array = weight_array.view(-1)
        #logger.debug("weight_array: {}".format(weight_array))
        return weight_array
    
    @jaxtyped(typechecker=typechecker)
    def get_dataset_w(
        self, 
        states:List[torch.Tensor], 
        actions:List[torch.Tensor], 
        eval_policy:BasePolicy
        )->Tuple[WeightTensor, WeightTensor]:
        """_summary_

        Args:
            states (List[torch.Tensor]): List of input Tensors of dimensions
            (traj_length, number of states)
            actions (List[torch.Tensor]): List of input Tensors of dimensions
                (traj_length, number of actions). Note, this is likely 
                (traj_length,1) if for example a discrete action space has been 
                flattened from [0,1]^2 to [0,1,2,3]
            eval_policy (BasePolicy): Policy class defining the target policy to be 
                evaluated

        Returns:
            Tuple[torch.Tensor]: Tuple of tensors where:
                weight_res is a (# trajectories, max(traj_length)) Tensor. 
                weight_res[i,j] defines the jth timestep propensity weight for 
                the ith trajectory
                weight_msk is a (# trajectories, max(traj_length)) binary 
                Tensor. weight_msk[i,j] defines whether the jth timestep of the
                ith trajectory was observed
        """
        assert len(states) == len(actions)
        #assert isinstance(eval_policy, BasePolicy)
        # weight_res = torch.zeros(size=(len(states),h))
        # weight_msk = torch.zeros(size=(len(states),h))
        weight_res_lst:List[torch.Tensor] = []
        weight_msk_lst:List[torch.Tensor] = []
        h = 0
        for i, (s,a) in enumerate(zip(states, actions)):
            weight = self.get_traj_w(
                states=s, 
                actions=a, 
                eval_policy=eval_policy
                )
            assert len(weight.shape) == 1 
            #weight_res_lst[i,:len(weight)] = weight
            weight_res_lst.append(weight)
            __h = len(weight)
            #weight_msk_lst[i,:len(weight)] = 
            weight_msk_lst.append(torch.ones(size=[__h]))
            h = max(h,__h)
        # pad all tensors to have same length
        weight_res_lst = [
            pad(x, pad=(0, h - x.numel()), mode='constant', value=0) 
            for x in weight_res_lst
            ]
        weight_res = torch.stack(weight_res_lst)
        # stack them
        weight_msk_lst = [
            pad(x, pad=(0, h - x.numel()), mode='constant', value=0) 
            for x in weight_msk_lst
            ]
        weight_msk = torch.stack(weight_msk_lst)
        return weight_res, weight_msk
    
    def update(
        self, 
        states:List[torch.Tensor], 
        actions:List[torch.Tensor], 
        eval_policy:BasePolicy
        ):
        _is_weights, _weight_msk = self.get_dataset_w(
            states=states, actions=actions, eval_policy=eval_policy)
        check_array_dim(_is_weights,2)
        check_array_dim(_weight_msk,2)
        len_act = len(actions)
        len_eval_act = len(eval_policy.policy_actions)
        _msg = f"""
        Actions have length: {len_act}. 
        Evalutaion policy predicted actions have length: {len_eval_act}
        """
        _equal_len_tst = len_act == len_eval_act
        _none_test = len_eval_act == 0
        assert _equal_len_tst or _none_test, _msg
        self.policy_actions = eval_policy.policy_actions
        self.is_weights = _is_weights
        self.weight_msk = _weight_msk

    def flush(self):
        self.is_weights = torch.empty(0)
        

class ImportanceSampler(metaclass=ABCMeta):
    
    def __init__(self, is_weight_calc:ISWeightCalculator) -> None:
        self.is_weight_calc = is_weight_calc
        self.traj_is_weights = torch.empty(0)
        
    
    def update(self):
        self.traj_is_weights = self.get_traj_weight_array(
            is_weights=self.is_weight_calc.is_weights,
            weight_msk=self.is_weight_calc.weight_msk
        )
    
    def flush(self):
        self.traj_is_weights = torch.empty(0)
    
    @abstractmethod
    def get_traj_weight_array(
        self, is_weights:torch.Tensor, weight_msk:torch.Tensor
        )->torch.Tensor:
        """Performs additional calculations on the weights i.e. for per 
        decision. Weights are products up to the current timestep

        Args:
            is_weights (torch.Tensor): Tensor of dimension 
                (# trajectories, max(traj_length))
            weight_msk (torch.Tensor): Tensor of dimension 
                (# trajectories, max(traj_length))
            
        Returns: 
            torch.Tensor: Transformed weight array of dimension 
                (# trajectories, max(traj_length))
        """
        pass
        
    
class VanillaIS(ImportanceSampler):
    
    @jaxtyped(typechecker=typechecker)
    def get_traj_weight_array(
        self, 
        is_weights:WeightTensor, 
        weight_msk:WeightTensor
        )->WeightTensor:
        assert isinstance(is_weights,torch.Tensor)
        assert isinstance(weight_msk,torch.Tensor)
        __orig_dim = is_weights.shape
        # Convert missing timesteps in trajectories from 0 to 1 otherwise prod
        # will be 0 for trjectories without the max number of timesteps
        is_weights[weight_msk==0] = 1
        is_weights = torch.prod(is_weights, dim=1, keepdim=True)
        is_weights = is_weights.expand(__orig_dim)
        # Convert missing timesteps in trajectories back to 0
        is_weights = is_weights*weight_msk
        return is_weights

class PerDecisionIS(ImportanceSampler):
    
    @jaxtyped(typechecker=typechecker)
    def get_traj_weight_array(
        self, 
        is_weights:WeightTensor, 
        weight_msk:WeightTensor
        )->WeightTensor:
        assert isinstance(is_weights,torch.Tensor)
        assert isinstance(weight_msk,torch.Tensor)
        __orig_dim = is_weights.shape
        is_weights = torch.cumprod(is_weights, dim=1)
        is_weights = is_weights*weight_msk
        check_array_shape(is_weights, __orig_dim)
        return is_weights


class ISWeightOrchestrator(ISWeightCalculator):
    
    __is_fact_lkp = {
        "vanilla": VanillaIS,
        "per_decision": PerDecisionIS
    }
    
    def __init__(self, *args, behav_policy:BasePolicy) -> None:
        super().__init__(behav_policy=behav_policy)
        self.is_samplers:Dict[str,ImportanceSampler] = {}
        for arg in args:
            try:
                self.is_samplers[arg] = self.__is_fact_lkp[arg](
                    is_weight_calc=self)
            except KeyError as e:
                logger.error("Accepted samplers are: {}".format(
                    list(self.__is_fact_lkp.keys())))
                
    def __len__(self):
            return len(self.is_samplers)

    def __getitem__(self, idx):
        return self.is_samplers[idx]
    
    def update(
        self, 
        states:List[torch.Tensor], 
        actions:List[torch.Tensor], 
        eval_policy:BasePolicy
        ):
        super().update(states=states, actions=actions, eval_policy=eval_policy)
        for sampler in self.is_samplers.keys():
            self.is_samplers[sampler].update()