# State inference via permutations in Hidden Markov Models


## Abstract

This package provides functionality for inference of the initial state of a
Hidden Markov Model (HMM), when we have access to permutations of the underlying states.
We provide both analytical calculations to compute the probability of correct inference,
and functionality for Monte Carlo computations. Further details are provided in
arxiv:xxxx.xxxxxx

## Project Status

*Maintenance only*

This software is mostly intended for research purposes, and
is not under active development.

## Testing summary

*Partial*

Some tests are provided, but are by no means comprehensive.
Unit tests for many functions are given in the tests/ directory,
the only integration tests done were the examples in the
example_scripts/ directory.

## Installation

This package uses python 3. If necessary, please install from [their website](https://www.python.org/downloads/).

Install the dependencies:

    git clone https://gitlab.nist.gov/gitlab/sng13/bayes_perm_hmm.git
    cd bayes_perm_hmm
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    
## Getting Started

A [Hidden Markov Model](https://en.wikipedia.org/wiki/Hidden_Markov_model) (HMM) is a Markov chain obscured by noise. 
Here we are interested in the task of identifying the initial state of an HMM, with the additional capability of 
permuting the underlying state space after each observation.

Here is a minimal code snippet showing the usage of the `PermutedDiscreteHMM`. The usage is nearly identical to that of 
the `pyro.distributions.DiscreteHMM`.

```python
import torch
import pyro.distributions as dist
from perm_hmm.models.hmms import PermutedDiscreteHMM
from perm_hmm.policies.min_tree import MinEntPolicy

initial_log_probs = torch.tensor([.5, .5]).log()
log_transition_matrix = torch.tensor([[.5, .5], [.1, .9]]).log()
outcome_probabilities = torch.tensor([.2, .3])
outcome_distribution = dist.Bernoulli(outcome_probabilities)
hmm = PermutedDiscreteHMM(initial_log_probs, log_transition_matrix, outcome_distribution)
possible_perms = torch.tensor([[0, 1], [1, 0]])
perm_policy = MinEntPolicy(possible_perms, hmm, save_history=True)
num_steps = 5
num_runs = 100
states, data = hmm.sample((num_runs, num_steps), perm_policy=perm_policy)
perms = perm_policy.perm_history
calc_history = perm_policy.calc_history
```

The `MinEntPolicy` is a `PermPolicy` that implements an algorithm for selecting permutations. The `MinEntPolicy` 
chooses the permutation that minimizes the posterior initial state entropy, looking forward one step.

After obtaining samples, we want to classify them. We are most interested in the 
[Maximum a Posteriori](https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation) (MAP) classifier for permutations,
here implemented in the `PermClassifier` class.

```python
from perm_hmm.classifiers.perm_classifier import PermClassifier
classifier = PermClassifier(hmm)
classifications = classifier.classify(data, perms)
```

Next, we want to use the classifications to compute some summary statistics. This is implemented in the postprocessing
module

```python
from perm_hmm.postprocessing import EmpiricalPostprocessor
post = EmpiricalPostprocessor(states, classifications)
rate_dict = post.misclassification_rate(confidence_level=.95)
rate, lower, upper = rate_dict[b'rate'], rate_dict[b'lower'], rate_dict[b'upper']
```

Further examples of usage of this package are given in the `example_scripts`
directory. A simple example to start with is given in `example_scripts/exhaustive_three_states.py`.
One can run that example with 
```shell
source venv/bin/activate
cd example_scripts
python ./exhaustive_three_states
```

Another example is given in `example_scripts/beryllium_plot.py`. One can run this with
```shell
source venv/bin/activate
cd example_scripts
python ./beryllium_plot.py
```


## Module listing

These modules are central to the idea of the package:
- `models`: Contains the permuted hmm class.
- `policies`: Contains methods for selecting permutations.

These modules provide functionality for the "interrupted" classifier:
- `classifiers.interrupted`: Contains the `InterruptedClassifier` class
- `training.interrupted_training`: Methods used to learn the parameters of the `InterruptedClassifier`

These modules provide methods and classes that use the `PermutedDiscreteHMM`
and the `InterruptedClassifier` in the context of inferring initial states:
- `simulator`: Simulates an experiment where data is generated from a `PermutedDiscreteHMM`,
    then classified with a MAP classifier.
  
- `postpocessing`: Provides methods which take as input classifications and outputs
    misclassification rates. Uses the `loss_functions` module.
  
- `rate_comparisons`: High level functions that wrap all the objects together
    to compare misclassification rates of the different classifiers.
  
These modules provide miscellaneous functionality:
- `util`: Utility functions
- `return_types`: Provides `NamedTuple`s that are returned by the various objects


## Documentation

Build the docs using sphinx-apidoc:

    pip install Sphinx
    cd docs
    make html   

Docs are then in the `docs/_build/html` folder, you can open the pages in your favorite
browser.

## Support

I will not have very much time to work on this project in the
future, so support will be minimal. 


## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/usnistgov/perm_hmm/CONTRIBUTING.md) 
for details on our code of conduct, and the process for submitting pull requests to us.

## Authors & Main Contributors

Shawn Geller is the primary author of this project.

See also the list of [contributors](https://github.com/usnistgov/perm_hmm/contributors) who participated in this project.

## Related Work

[POMDPs.jl](https://github.com/JuliaPOMDP/POMDPs.jl)

[pomegranate](https://pomegranate.readthedocs.io/en/latest/)

[hmmlearn](https://github.com/hmmlearn/hmmlearn)

[pyro](https://github.com/pyro-ppl/pyro)

[torch](https://github.com/pytorch/pytorch)


## Copyright

To see the latest statement, please visit:
https://www.nist.gov/director/copyright-fair-use-and-licensing-statements-srd-data-and-software

Add a directory where you put licenses of any dependent software or if
the project you are working on is a fork.

Also see the [LICENSE.md](https://github.com/your/project/LICENSE.md)

## Acknowledgments

Discussions and theoretical support provided by Emanuel Knill, Scott Glancy,
and Daniel Cole.
We thank Dietrich Leibfried for introducing us in the early days of ion trap quantum computing to the
idea of adaptively chosen pulses for improving measurement fidelity. We also thank Zachary
Sunberg for discussions on the POMDP formalism, and. We thank Giorgio Zarantonello for
computations involving the transition rates in Beryllium. We thank Mohammad Alhejji, Alexander Kwiatkowski, Arik Avagyan, 
Akira Kyle, and Stephen Erickson for helpful suggestions and comments.


## Contact

Shawn Geller: shawn.geller@colorado.edu
Scott Glancy: scott.glancy@nist.gov
Emanuel Knill: emanuel.knill@nist.gov
