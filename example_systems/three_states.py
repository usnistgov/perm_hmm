r"""
This module implements a simple three state model shown in the figure. The
circles on the left represent states, while the squares on the right are
outputs.

.. image:: _static/three_state_model.svg
"""
import numpy as np
import torch
import pyro.distributions as dist
from perm_hmm.util import ZERO, log1mexp
from perm_hmm.models.hmms import PermutedDiscreteHMM


def three_state_params(a, b, c=None, d=None):
    """Gives a list of parameters for the three state model.

    :return: A tuple of (initial state probabilities (uniform),
        transition probabilities,
        output probabilities)
    """
    num_states = 3
    if c is None:
        c = b
    if d is None:
        d = a
    initial_probs = np.ones(num_states)/num_states
    output_probs = np.array([[1-a, a, 0.], [(1-d)/2, d, (1-d)/2], [0., a, (1-a)-0.]])
    transition_probs = np.array([[1-b-0., b, 0.],[(1-c)/2, c, (1-c)/2],[0., b, 1-b-0.]])
    return initial_probs, transition_probs, output_probs



def three_state_log_params(log_a, log_b, log_c=None, log_d=None):
    """Log parameters. Inputs are log as well.

    :return: A tuple of (log initial state probabilities (uniform),
        log transition probabilities,
        log output probabilities)
    """
    num_states = 3
    if log_c is None:
        log_c = log_b
    if log_d is None:
        log_d = log_a
    initial_logits = np.zeros((num_states,)) - np.log(num_states)
    nota = log1mexp(log_a)
    notb = log1mexp(log_b)
    notc = log1mexp(log_c)
    notd = log1mexp(log_d)
    logzero = np.log(ZERO)
    output_logits = np.array([
        [nota, log_a, logzero],
        [notd - np.log(2), log_d, notd - np.log(2)],
        [logzero, log_a, nota]
    ])
    transition_logits = np.array([
        [notb, log_b, logzero],
        [notc - np.log(2), log_c, notc - np.log(2)],
        [logzero, log_b, notb]
    ])
    return initial_logits, transition_logits, output_logits


def three_state_hmm(log_a, log_b, log_c=None, log_d=None):
    r"""Gives a Permuted discrete HMM for the inputs.

    :return: A PermutedDiscreteHMM for the input parameters.
    """
    initial_logits, transition_logits, output_logits = [torch.from_numpy(x) for x in three_state_log_params(log_a, log_b, log_c, log_d)]
    observation_dist = dist.Categorical(logits=output_logits)
    hmm = PermutedDiscreteHMM(
        initial_logits,
        transition_logits,
        observation_dist,
    )
    return hmm
