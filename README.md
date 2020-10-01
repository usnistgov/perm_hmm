# State inference via permutations in Hidden Markov Models

# Abstract

Computes misclassification rates for various repeated measurement schemes,
using a Hidden Markov Model. By implementing permutations in the middle
of a repeated measurement, one can obtain higher accuracy in the 
inference of the initial state in the Markov Chain.
This software gives tools for computing misclassification rates for
such a scheme, as well as comparing to simpler repeated measurement
schemes.

# Installation

This package uses python 3. If necessary, please install from [their website](https://www.python.org/downloads/).

Install from source:

    git clone https://gitlab.nist.gov/gitlab/sng13/bayes_perm_hmm.git
    cd bayes_perm_hmm
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
    pip install .
    
Once you're done using the virtual environment, deactivate it with `deactivate`

Example scripts are in the `example_scripts` directory. Try using one of them:

    python example_scripts/beryllium_example.py -e 1e-7 -o beryllium.pt 5
    
# Documentation

Build the docs using sphinx-apidoc:

    pip install -U Sphinx
    sphinx-apidoc -f -o docs .
    cd docs
    make html   

Docs are then in the `docs/_build/html` folder, you can open the pages in your favorite
browser.

# Paper

See arXiv:xxxx.xxxxx for further details.

# Description


Given a set of measurements and a single copy of a state, what is the optimal measurement strategy to minimize
one-shot misclassification error? For perfect measurements, clearly we cannot extract information after the first
observation. However, for imperfect measurements, there is an effective transition model for the post-measurement state.
We can model this as a path through discrete state space, where each measurement induces a jump between the pure states
with some probability. The new probability of measuring a state is given by the 
