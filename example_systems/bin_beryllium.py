import os
import argparse
from itertools import combinations
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.special import logsumexp
import pyro.distributions as dist
from pyro.distributions import Categorical
from perm_hmm.models.hmms import ExpandedHMM
from perm_hmm.simulator import HMMSimulator
from perm_hmm.util import kl_divergence
from adapt_hypo_test.two_states.util import log1mexp
from example_systems import beryllium as be
from example_systems.beryllium import DARK_STATE, BRIGHT_STATE, N_STATES, dimensionful_gamma, expanded_transitions, expanded_initial, expanded_outcomes


def save_hist_plot(fig, savedir):
    plt.sca(fig.gca())
    fn = os.path.join(savedir, "hist.svg")
    plt.savefig(fn)
    return fn


def save_histograms(binned, savedir):
    fn = os.path.join(savedir, "histograms.npz")
    np.savez(
        fn,
        binned=binned,
    )
    return fn


def plot_binned(binned, bin_edges, labels=None, fig=None):
    if fig is None:
        fig = plt.figure()
    if labels is None:
        labels = ["Dark", "Bright"]
    ax = fig.gca()
    for b, l in zip(binned, labels):
        ax.bar(bin_edges[:-1], np.exp(b), np.diff(bin_edges), align="edge", alpha=.5, label=l)
    plt.xlabel("Number of photons")
    plt.ylabel("Probability of detection")
    return fig


def plot_unbinned(hists, labels=None, fig=None):
    if fig is None:
        fig = plt.figure()
    if labels is None:
        labels = ["Dark", "Bright"]
    ax = fig.gca()
    for h, l in zip(hists, labels):
        ax.bar(np.arange(hists.shape[-1]), np.exp(h), align="edge", alpha=.5, label=l)
    return fig


def bin_histogram(base_hist, bin_edges):
    new_hist = []
    for i in range(len(bin_edges)-1):
        new_hist.append(logsumexp(base_hist[..., bin_edges[i]:bin_edges[i+1]], axis=-1))
    return np.stack(new_hist, axis=-1)


def find_optimal_binning(base_hists, value_func, num_bins, return_opt=False):
    max_phot = base_hists.shape[-1]
    max_value = None
    best_bin_edges = None
    best_binned = None
    for bin_edges in combinations(np.arange(max_phot-2)+1, num_bins-1):
        bin_edges = np.concatenate((np.array([0]), bin_edges, np.array([max_phot])))
        binned = bin_histogram(base_hists, bin_edges)
        value = value_func(bin_edges)
        if (max_value is None) or (value > max_value):
            max_value = value
            best_bin_edges = bin_edges
            best_binned = binned
    retval = {
        b"binned_bright_dark": best_binned,
        b"bin_edges": best_bin_edges,
    }
    if return_opt:
        retval[b"max_value"] = max_value
    return retval


def unbinned_hists(integration_time, max_photons=10):
    hists = be.log_prob_n_given_l(np.arange(max_photons), integration_time)
    total_weight = logsumexp(hists, -2)
    hists[-1] = logsumexp(np.stack([hists[-1, :], log1mexp(total_weight)], axis=-2), axis=-2)
    hists = hists[..., [DARK_STATE, BRIGHT_STATE]]
    hists = hists.transpose()
    return hists


def bin_bright_dark(hists, num_bins, value_func=None, return_opt=False):
    if value_func is None:
        def value_func(bin_edges):
            bright_dark_hists = bin_histogram(hists, bin_edges)
            return kl_divergence(bright_dark_hists[0, :], bright_dark_hists[1, :]) + kl_divergence(bright_dark_hists[1, :], bright_dark_hists[0, :])
    return find_optimal_binning(hists, value_func, num_bins, return_opt=return_opt)


def bin_all(hists, num_bins, value_func=None, return_opt=False):
    retval = bin_bright_dark(hists, num_bins, value_func=value_func, return_opt=return_opt)
    bin_edges = retval[b"bin_edges"]
    binned_hists = bin_histogram(hists, bin_edges)
    retval.pop(b"binned_bright_dark")
    retval[b"binned_hists"] = binned_hists
    return retval


def bin_hmm_from_bin_edges(time, bin_edges, max_photons):
    num_bins = len(bin_edges) - 1
    ltm = expanded_transitions(time, k=max_photons)
    ltm = ltm.reshape((N_STATES, max_photons, N_STATES, max_photons))
    binned_transitions = bin_histogram(ltm, bin_edges)[:, np.arange(num_bins), ...].reshape((N_STATES*num_bins, N_STATES*num_bins))  # Doesn't matter which "previous outcome" slice we take, as long as it has the right size.
    binned_initial = expanded_initial(k=num_bins)
    binned_outcomes = expanded_outcomes(k=num_bins)
    observation_dist = Categorical(logits=torch.from_numpy(binned_outcomes))
    hmm = ExpandedHMM(torch.from_numpy(binned_initial), torch.from_numpy(binned_transitions), observation_dist)
    return hmm


def log_class_prob(model, steps):
    sim = HMMSimulator(model)
    ep = sim.all_classifications(steps)
    log_misclass_rate = ep.log_misclassification_rate()
    return log1mexp(log_misclass_rate)


def make_value_func(time, steps, max_photons):
    def value_func(bin_edges):
        model = bin_hmm_from_bin_edges(time, bin_edges, max_photons)
        return log_class_prob(model, steps)
    return value_func


def get_optimal_bin_edges(time, max_photons=10, num_bins=4, value_func=None, verbosity=0):
    base_hists = unbinned_hists(time, max_photons=max_photons)
    return bin_all(base_hists, num_bins=num_bins, value_func=value_func, return_opt=bool(verbosity))


def binned_hmm_class_value(time, max_photons=10, num_bins=4, steps=3, verbosity=0):
    optimal_binning = get_optimal_bin_edges(time, max_photons=max_photons, num_bins=num_bins, value_func=make_value_func(time, steps, max_photons), verbosity=verbosity)
    bin_edges = optimal_binning[b"bin_edges"]
    binned_hmm = bin_hmm_from_bin_edges(time, bin_edges, max_photons)
    return {
        b"hmm": binned_hmm,
        b"optimal_binning": optimal_binning,
    }


def bin_hmm(time, max_photons=10, num_bins=4, value_func=None, verbosity=0):
    r"""Given an integration time, bins the output distributions such that the
    symmetrized divergence from the bright state to the dark state is maximal,
    then returns everything used to compute that, along with the resulting HMM.

    :param time:
    :return:
    """
    base_hists = unbinned_hists(time, max_photons=max_photons)
    opt_bin = bin_all(base_hists, num_bins, value_func=value_func, return_opt=bool(verbosity))
    bin_edges = opt_bin[b"bin_edges"]
    ltm = expanded_transitions(time, k=max_photons)
    ltm = ltm.reshape((N_STATES, max_photons, N_STATES, max_photons))
    binned_transitions = bin_histogram(ltm, bin_edges)[:, np.arange(num_bins), ...].reshape((N_STATES*num_bins, N_STATES*num_bins))  # Doesn't matter which "previous outcome" slice we take, as long as it has the right size.
    binned_initial = expanded_initial(k=num_bins)
    # binned_initial = bin_histogram(base_initial, bin_edges)
    binned_outcomes = expanded_outcomes(k=num_bins)
    observation_dist = dist.Categorical(logits=torch.from_numpy(binned_outcomes))
    hmm = ExpandedHMM(torch.from_numpy(binned_initial), torch.from_numpy(binned_transitions), observation_dist)
    retval = {
        b"hmm": hmm,
        b"base_hists": base_hists,
        b"optimal_binning": opt_bin,
    }
    return retval


def main(dimensionful_integration_time, num_bins, max_phot=10, savedir=None, save=False):
    if num_bins > max_phot:
        raise ValueError("Too many bins")
    integration_time = dimensionful_integration_time * dimensionful_gamma
    hists = unbinned_hists(integration_time, max_photons=max_phot)
    bin_dict = bin_bright_dark(hists, num_bins, return_opt=True)
    binned = bin_dict[b"binned_bright_dark"]
    bin_edges = bin_dict[b"bin_edges"]
    opt_divergence = bin_dict[b"max_value"]
    fig = plot_binned(binned, bin_edges)
    fig = plot_unbinned(hists, fig=fig)
    if save:
        if savedir is None:
            savedir = os.getcwd()
        _ = save_hist_plot(fig, savedir)
        _ = save_histograms(binned, savedir)
    plt.show()
    unbinned_divergence = kl_divergence(hists[0, :], hists[1, :]) + kl_divergence(hists[1, :], hists[0, :])
    print("Time: {}, Bins: {}, Photons: {}, Binned divergence: {}, Unbinned divergence: {}".format(dimensionful_integration_time, num_bins, max_phot, opt_divergence, unbinned_divergence))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "integration_time",
        metavar="integration-time",
        help="The amount of time to integrate the collection of photons for,"
             "in units of 1/(2 * pi * 19.4 * 10 ** 6 Hz)",
        type=float,
    )
    parser.add_argument(
        "num_bins",
        metavar="num-bins",
        type=int,
        default=5,
    )
    parser.add_argument(
        "max_phot",
        metavar="max-phot",
        type=int,
        default=5,
    )
    args = parser.parse_args()
    main(args.integration_time, args.num_bins, max_phot=args.max_phot)
