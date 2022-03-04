r"""Log costs to be used with the
:py:class:`~perm_hmm.policies.min_tree.MinTreePolicy` class.
"""
import torch


def log_initial_entropy(log_probs: torch.Tensor):
    """
    Calculates the log of the initial state posterior entropy from log_probs, with dimensions
    -1: s_k, -2: s_1

    :param log_probs:
    :return:
    """
    inits = log_probs.logsumexp(-1)
    return ((-inits).log() + inits).logsumexp(-1)


def log_renyi_entropy(log_probs: torch.Tensor, alpha: float):
    """
    Calculates the log of initial state posterior Renyi entropy from log_probs, with dimensions
    -1: s_k, -2: s_1

    :param log_probs:
    :param alpha:
    :return:
    """
    inits = log_probs.logsumexp(-1)
    return ((inits*alpha).logsumexp(-1)/(1-alpha)).log()


def log_min_entropy(log_probs: torch.Tensor):
    """
    Calculates the log of minimum initial state posterior entropy from log_probs, with dimensions
    -1: s_k, -2: s_1

    :param log_probs:
    :return:
    """
    inits = log_probs.logsumexp(-1)
    return (((-inits).log()).min(-1))[0]


def min_entropy(log_probs):
    inits = log_probs.logsumexp(-1)
    return ((inits.max(-1))[0])
