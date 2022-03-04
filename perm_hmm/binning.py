import warnings
import torch
from pyro.distributions import Categorical
from itertools import combinations
from perm_hmm.models.hmms import ExpandedHMM, DiscreteHMM
from perm_hmm.simulator import HMMSimulator


def bin_histogram(base_hist, bin_edges):
    r"""Given a histogram, bins it using the given bin edges.

    Bin edges are left inclusive, right exclusive, so that::

        bin_histogram(base_hist, bin_edges)[..., i] = base_hist[..., bin_edges[i]:bin_edges[i+1]].sum()

    :param base_hist: The histogram to bin. This is a probability distribution,
        so that ``torch.allclose(base_hist.sum(-1), 1.)``
    :param bin_edges: The edges of the bins. Should be in increasing order.
    :return: Binned histograms, shape ``base_hist.shape[:-1] + (bin_edges.shape[-1]-1,)``
    """
    if not torch.allclose(torch.sum(base_hist, dim=-1).float(), torch.tensor(1.).float()):
        warnings.warn("The input histogram is not normalized.")
    n = base_hist.shape[-1]
    new_hist = []
    for i in range(len(bin_edges) - 1):
        new_hist.append(
            torch.sum(base_hist[..., bin_edges[i]:bin_edges[i + 1]], dim=-1))
    new_hist = torch.stack(new_hist, dim=-1)
    if not torch.allclose(torch.sum(new_hist, dim=-1).float(), torch.tensor(1.).float()):
        warnings.warn("The bin edges are such that the new histogram is not normalized. "
                         "Maybe the edges 0 and {} weren't included?".format(n))
    return new_hist

def bin_log_histogram(base_log_hist, bin_edges):
    r"""Given a histogram, bins it using the given bin edges.

    Bin edges are left inclusive, right exclusive, so that::

        bin_log_histogram(base_hist, bin_edges)[..., i] = base_hist[..., bin_edges[i]:bin_edges[i+1]].logsumexp()

    :param base_hist: The histogram to bin. This is a probability distribution,
        so that ``torch.allclose(base_hist.logsumexp(-1), 0.)``
    :param bin_edges: The edges of the bins. Should be in increasing order.
    :return: Binned histograms, shape ``base_hist.shape[:-1] + (bin_edges.shape[-1]-1,)``
    """
    if not torch.allclose(torch.logsumexp(base_log_hist, dim=-1).float(), torch.tensor(0.).float(), atol=1e-7):
        warnings.warn("The input histogram is not normalized.")
    new_hist = []
    for i in range(len(bin_edges) - 1):
        new_hist.append(
            torch.logsumexp(base_log_hist[..., bin_edges[i]:bin_edges[i + 1]], dim=-1))
    new_hist = torch.stack(new_hist, dim=-1)
    if not torch.allclose(torch.logsumexp(new_hist, dim=-1).float(), torch.tensor(0.).float(), atol=1e-7):
        warnings.warn(
            "The bin edges are such that the new histogram is not normalized. "
            "Maybe the edges 0 and {} weren't included?".format(new_hist.shape[-1]))
    return new_hist


def binned_hmm(hmm, bin_edges):
    r"""Assuming a categorical output distribution, finds the binned version of
    the HMM.

    Applies ``bin_histogram`` to the output distributions if probs is specified,
    otherwise applies ``bin_log_histogram``.
    """
    if torch.any(hmm.observation_dist._param < 0):
        # sum log_probs
        base_log_hist = hmm.observation_dist._param
        observation_dist = Categorical(logits=bin_log_histogram(base_log_hist, bin_edges))
    else:
        observation_dist = Categorical(probs=bin_histogram(hmm.observation_dist._param, bin_edges))
    return type(hmm)(
        initial_logits=hmm.initial_logits,
        transition_logits=hmm.transition_logits,
        observation_dist=observation_dist,
    )


def binned_expanded_hmm(expanded_hmm, bin_edges):
    r"""Given an HMM with an expanded state space used to account for outcome
    dependence, bins the HMM using the bin edges.
    """
    num_bins = len(bin_edges) - 1
    ltm = expanded_hmm.transition_logits
    n, o = expanded_hmm.i_to_lo(ltm.shape[-1]-1)
    n += 1
    o += 1
    tm = ltm.reshape((n, o, n, o))
    binned_transitions = bin_log_histogram(tm, bin_edges)[:, torch.arange(num_bins), ...].reshape((
        n * num_bins,
        n * num_bins))  # Doesn't matter which "previous outcome" slice we take, as long as it has the right size.
    binned_initial = bin_log_histogram(expanded_hmm.initial_logits.reshape((n, o)), bin_edges).reshape((-1,))
    if torch.any(expanded_hmm.observation_dist._param < 0):
        # sum log_probs
        base_log_hist = expanded_hmm.observation_dist._param
        base_log_hist = base_log_hist.reshape((n, o, o))[:, bin_edges[:-1]]
        binned_outputs = bin_log_histogram(base_log_hist, bin_edges).reshape((n*num_bins, num_bins))
        observation_dist = Categorical(logits=binned_outputs)
    else:
        base_hist = expanded_hmm.observation_dist._param
        base_hist = base_hist.reshape((n, o, o))[:, bin_edges[:-1]]
        binned_outputs = bin_histogram(base_hist, bin_edges).reshape((n*num_bins, num_bins))
        observation_dist = Categorical(probs=binned_outputs)
    hmm = type(expanded_hmm)(binned_initial, binned_transitions, observation_dist)
    return hmm


def generate_infidelity_cost_func(hmm, num_steps):
    r"""Generates a function that takes bin edges and returns the infidelity of
    a binned hmm after collecting ``num_steps`` steps of data.
    """
    is_expanded = isinstance(hmm, ExpandedHMM)

    def infidelity_cost_func(bin_edges):
        if is_expanded:
            binned = binned_expanded_hmm(hmm, bin_edges)
        else:
            binned = binned_hmm(hmm, bin_edges)
        sim = HMMSimulator(binned)
        ep = sim.all_classifications(num_steps)
        return ep.log_misclassification_rate()

    return infidelity_cost_func


def optimally_binned_consecutive(hmm, num_bins, cost_func=None, steps=2):
    r"""Given an hmm, finds the optimal binning by exhaustively searching over all possible
    bin edges.

    The bin edges dictate which consecutive outcomes to include in a bin.

    WARNING: This method is slow. The complexity grows as :math:`O(Y^n)`, where
    :math:`Y` is the number of outcomes of the unbinned histogram, and :math:`n`
    is the number of bins to use.

    By default, uses the infidelity at ``steps`` number of steps of the binned hmm as the
    cost function. This can potentially be expensive to compute, so use a different cost function when necessary.

    :param cost_func: A function that takes bin edges and returns a cost. Defaults
        to ``generate_infidelity_cost_func(hmm, steps)``.
    :param steps: Only used to make a cost function, when that is not specified.
    :return: The optimal bin edges, and the minimal cost.
    """
    max_observation = hmm.enumerate_support(False).reshape((-1,)).shape[-1]
    if num_bins > max_observation:
        raise ValueError("Too many bins for number of outcomes. Asked for "
                         "{} bins for {} outcomes.".format(num_bins, max_observation))
    if num_bins < 2:
        raise ValueError("Can't have fewer than 2 bins. Asked for {} bins.".format(num_bins))
    if cost_func is None:
        cost_func = generate_infidelity_cost_func(hmm, steps)
    minimizing_edges = None
    min_cost = None
    for bin_edges in combinations(torch.arange(max_observation - 2)+1, num_bins-1):
        bin_edges = torch.concat((torch.tensor([0], dtype=int), torch.tensor(bin_edges), torch.tensor([max_observation], dtype=int)))
        cost = cost_func(bin_edges)
        if (min_cost is None) or (cost < min_cost):
            min_cost = cost
            minimizing_edges = bin_edges
    return minimizing_edges, min_cost
