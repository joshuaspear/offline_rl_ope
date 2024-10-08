# offline rl ope: A Python package for off-policy evaluation of offline RL models with realworld data

> [!WARNING]
> - All IS methods implemented incorrectly in versions < 6.x
> - Unit testing currently only running in Python 3.11. 3.10 will be supported in the future
> - Not all functionality has been tested i.e., d3rlpy api and LowerBounds are still in beta

### Testing progress
- [x] components/
  - [x] ImportanceSampler.py (except runtime check for policy class (forthcoming))
  - [x] Policy.py
- [x] OPEEstimation
  - [x] IS.py
  - [x] utils.py
  - [x] DirectMethod.py
  - [x] DoublyRobust.py
- [x] Metrics
  - [x] EffectiveSampleSize.py
  - [x] ValidWeightsProp.py
- [x] PropensityModels
- [ ] LowerBounds
- [ ] api/d3rlpy


#### Overview
Basic unit testing has been implemented for all the core functionality of the package. The d3rlpy/api for importance sampling adds minimal additional functionality therefore, it is likely to function as expected however, no sepcific unit testing has been implemented! 

> [!IMPORTANT]
> * More documentation needs to be added however, please refer to examples/ for an illustration of the functionality
> * examples/static.py provides an illustration of the package being used for evaluation post training. Whilst the d3rlpy package is used for model training, the script is agnostic to the evaluation model used
> * examples/d3rlpy_training_api.py provides an illustration of how the package can be used to obtain incremental performance statistics during the training of d3rlpy models. It provides greater functionality to the native scorer metrics included in d3rlpy

### Description
* offline_rl_ope aims to provide flexible and efficient implementations of OPE algorithms for use when training offline RL models. The main audience is researchers developing smaller, non-distributed models i.e., those who do not want to use packages such as ray (https://github.com/ray-project/ray).
* The majority of the first phase of development has been on efficient implementations of importance sampling and integration with d3rlpy.

#### Main user assumptions of the project
* Users require smaller workloads
* Users should be able to easily implement new OPE methods using the importance sampling classes provided
* When developing models, users may want to run several different OPE methods e.g., per-decision IS, per-decision weight IS etc. These different methods fundamentally utilise the same set of IS weights. The library has been developed to efficiently cache these IS weights, significantly reducing the overhead of incremental OPE metrics whilst adding no overhead if only a single OPE method is required.

#### OPE methods
* Importance sampling:
  * Weighted
  * Per decision
  * Clipped
* Fitted Q-Evaluation (via d3rlpy)
* Doubly robust:
  * Weighted
  * Per decision
  * Clipped

#### Lower bounds
* HCOPE (via https://github.com/gonzfe05/HCOPE/tree/master)
* Bootstrapping

### Credits
* Credit to https://github.com/takuseno/d3rlpy for the implementation of FQE
* Credit to https://github.com/gonzfe05/HCOPE/tree/master for the implementation of HCOPE


### Installation
* PyPi: https://pypi.org/project/offline-rl-ope/
* To install from sourse using pip, clone this repository and run ```pip install .``` in the top level of the repo.

### Citation
When using this package in your work, please cite the following:
```
@misc{Spear2024, 
  author = {Joshua Spear},
  title = {offline_rl_ope: A Python package for off-policy evaluation of offline RL models with realworld data},
  year = {2024},
  month = {9}, 
  url = {https://github.com/joshuaspear/offline_rl_ope},  
}
```

### Future work
* Async/multithread support
* Additional estimators:
  * DualDICE
  * MAGIC
  * State importance sampling
* APIs
  * Add additional apis e.g. for stable baselines


### Tips
#### Debugging importance ratios
If importance sampling based methods are evaluating to 0, consider visualising the importance ratios at different stages. All IS based estimators require an object of types ```ISWeightCalculator``` to be defined (this could be a ```ISWeightCalculator``` object or a subclass e.g., ```ISWeightOrchestrator```). In any case, visualising the ```is_weights``` attribute of this object will provide insight regarding the course of NaN's.
* ```NaN``` values occur when the behaviour policy and evaluation policy have probability 0. In both cases a small epsilon value could be used in place of probability 0. For deterministic evaluation policies, this is automatically included ```D3RlPyDeterministic```.
* ```Inf``` values can occur when the probability under the evaluation policy is greater than 0 whilst the behaviour policu probability is 0.

The different kinds of importance samples can also be visualised by querying the ```traj_is_weights``` attribute of a given ```ImportanceSampler``` object. If for example, vanilla importance sampling is being used and the samples are not ```NaN``` or ```Inf``` then visualising the ```traj_is_weights``` may provide insight. In particular, IS weights will tend to inifinity when the evaluation policy places large density on an action in comparison to the behaviour policy.

### Release log

#### 7.0.1
* Made the logging location for d3rlpy/FQE callback an optional parameter
* D3rlpy API now handles the use of observation and action scalers
* D3rlpy DirectMethod (D3rlpyQlearnDM) now handles automated reverse reward scaling
* Altered PVW metric to use the final weight in the trajectory

#### 7.0.0 (Major API release)
* Altered ISEstimator and OPEEstimatorBase APIs to depend on EmpiricalMeanDenomBase and WeightDenomBase
  * EmpiricalMeanDenomBase and WeightDenomBase seperatly define functions over the dataset value and weights of the individul trajectory weights, respectively. This allows a far greater number of estimators to be flexibly implemented
* Added api/StandardEstimators for IS and DR to allow for 'plug-and-play' analysis
* Altered discrete torch propensity model to use softmax instead of torch. Requires modelling both classes for binary classification however, improves generalisability of code

#### 6.0.0
* Updated PropensityModels structure for sklearn and added a helper class for compatability with torch
* Full runtime typechecking with jaxtyping
* Fixed bug with IS methods where the average was being taken twice
* Significantly simplified API, especially integrating Policy classes with propensity models
* Generalised d3rlpy API to allow for wrapping continuous policies with D3RlPyTorchAlgoPredict
* Added explicit stochastic policies for d3rlpy
* Introduced 'policy_func' which is any function/method which outputs type Union[TorchPolicyReturn, NumpyPolicyReturn]
* Simplified and unified ISCallback in d3rlpy/api using PolicyFactory
* Added 'premade' doubly robust estimators for vanilla DR, weighted DR, per-decision DR and weighted per-decision DR 

#### 5.0.1
* Fixed bug where GreedyDeterministic couldn't handle multi-dimensional action spaces
#### 5.0.0
* Correctly implemented per-decision weighted importance sampling
* Expanded the different types of weights that can be implemented based on:
  * http://proceedings.mlr.press/v48/jiang16.pdf: Per-decision weights are defined as the average weight at a given timepoint. This results in a different denominator for different timepoints. This is implemented with the following ```WISWeightNorm(avg_denom=True)```
  * https://scholarworks.umass.edu/cgi/viewcontent.cgi?article=1079&context=cs_faculty_pubs: Per-decision weights are defined as the sum of discounted weights across all timesteps. This is implemented with the following ```WISWeightNorm(discount=discount_value)```
  * Combinations of different weights can be easily implemented for example 'average discounted weights' ```WISWeightNorm(discount=discount_value, avg_denom=True)``` however, these do not necessaily have backing from literature.
* EffectiveSampleSize metric optinally returns nan if all weights are 0
* Bug fixes:
  * Fix bug when running on cuda where tensors were not being pushed to CPU
  * Improved static typing
#### 4.0.0
* Predefined propensity models including:
  * Generic feedforward MLP for continuous and discrete action spaces built in PyTorch
  * xGBoost for continuous and discrete action spaces built in sklearn
  * Both PyTorch and sklearn models can handle space discrete actions spaces i.e., a propensity model can be exposed to 'new' actions provided the full action space definition is provided at the training time of the propensity model
* Metrics pattern with:
  * Effective sample size calculation
  * Proportion of valid weights i.e., the mean proportion of weights between a min and max value across trajectories
* Refactored the BehavPolicy class to accept a 'policy_func' that aligns with the other policy classes
#### 3.0.3 
* 3.10 support
#### 3.0.2 
* PyPI release!
* Fixed bug in VanillaIS calculation where trajectories with less than the max number of samples were always being evaluated to 0
* Epsilon smoothing for deterministic evaluation policies
#### 3.0.1 
* Updated d3rlpy API to align with the 2.x release!
#### 2.3.0
* Corrected error when calculating weighted importance samples. The weight was defined as the sum rather than the mean of time t ratios
* Implemented smoothing for weighted importance sampling to prevents nan's when valid weights are 0. 
