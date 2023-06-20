# offline_rl_ope (BETA RELEASE)

**IMPORTANT: THIS IS A BETA RELEASE. FUNCTIONALITY IS STILL BEING TESTED** Feedback/contributions are welcome :) 

**IMPORTANT**:
* To use the d3rlpy api scorers, the following commit is required which is waiting approval https://github.com/takuseno/d3rlpy/pull/286
* More documentation needs to be added however, please refer to examples/ for an illustration of the functionality
* The current focus has been on discrete action spaces. Continuous action spaces are intended to be addressed at a later date

### Description
* Library for performing off policy evaluation, specifically for offline RL. 
* The library is designed to be applied standalone however, it also includes an api for d3rlpy to enable the different OPE methods to be used during training runs.

#### OPE methods
* Importance sampling:
 * Weighted
 * Per decision
 * Clipped
* Fitted Q-Evaluation (via d3rlpy)

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
