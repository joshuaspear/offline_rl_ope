# offline_rl_ope (BETA RELEASE)

**WARNING: Weighted importance sampling was incorrectly implemented in version 1.X.X**

**IMPORTANT: THIS IS A BETA RELEASE. FUNCTIONALITY IS STILL BEING TESTED** Feedback/contributions are welcome :) 

**IMPORTANT**:
* More documentation needs to be added however, please refer to examples/ for an illustration of the functionality
  * examples/static.py provides an illustration of the package being used for evaluation post training. Whilst the d3rlpy package is used for model training, the script is agnostic to the evaluation model used
  * examples/d3rlpy_training_api.py provides an illustration of how the package can be used to obtain incremental performance statistics during the training of d3rlpy models. It provides greater functionality to the native scorer metrics included in d3rlpy
* The current focus has been on discrete action spaces. Continuous action spaces are intended to be addressed at a later date

### Description
* offline_rl_ope aims to provide flexible and efficient implementations of OPE algorithms for use when training offline RL models. The main audience is researchers developing smaller, non-distributed models i.e., those who do not want to use packages such as ray (https://github.com/ray-project/ray).
* The majority of the first phase of development has been on efficient implementations of importance sampling and integration with d3rlpy.

#### Main user assumptions of the project
* Users require smaller workloads
* Users should be able to easily implment new OPE methods using the importance sampling classes provided
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
* This library is not currently available on pypi however, a version will be released soon.
* To install, clone/download the repository. Navtigate to the repository and run "pip install -e ."

### Future work
* Async/multithread support
* Additional estimators:
  * (Weighted) Doubly robust
  * MAGIC
* Continuous action spaces
