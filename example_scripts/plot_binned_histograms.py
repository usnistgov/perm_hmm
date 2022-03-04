import os
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)
import fire
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import torch
import pyro.distributions as dist

from example_systems.beryllium import dimensionful_gamma, expanded_transitions, expanded_initial, expanded_outcomes, N_STATES, DARK_STATE, BRIGHT_STATE
from perm_hmm.models.hmms import ExpandedHMM
from perm_hmm.binning import optimally_binned_consecutive, bin_log_histogram


def get_binned_histograms(time, num_bins, max_photons, steps):
    initial_logits = expanded_initial(max_photons)
    transition_logits = expanded_transitions(time, max_photons)
    observation_dist = dist.Categorical(logits=torch.from_numpy(expanded_outcomes(max_photons)))
    expanded_hmm = ExpandedHMM(
        torch.from_numpy(initial_logits),
        torch.from_numpy(transition_logits),
        observation_dist,
    )
    edges, min_infidelity = optimally_binned_consecutive(expanded_hmm, num_bins, steps=steps)
    unbinned_hists = logsumexp(transition_logits.reshape((N_STATES, max_photons, N_STATES, max_photons))[:, 0, :, :], axis=-2)
    return {
        "bin_edges": edges.numpy(),
        "unbinned_hists": unbinned_hists,
        "binned_hists": bin_log_histogram(torch.tensor(unbinned_hists), edges).numpy(),
    }


def plot_binned_histograms(
        data_directory=None,
):
    if data_directory is None:
        data_directory = os.getcwd()

    time = 5.39e-5 * dimensionful_gamma
    num_bins = 4
    max_photons = 15
    steps = 6
    data = get_binned_histograms(
        time,
        num_bins,
        max_photons,
        steps,
    )
    bin_edges = data["bin_edges"]
    unbinned_hist = data["unbinned_hists"]
    binned_hist = data["binned_hists"]
    fig = plt.figure()
    [[ax1, ax2], [ax3, ax4]] = fig.subplots(2, 2)
    ax1.bar(np.arange(max_photons), np.exp(unbinned_hist[DARK_STATE]), color="C0")
    ax1.set_title("Dark unbinned")
    ax3.bar(np.arange(max_photons), np.exp(unbinned_hist[BRIGHT_STATE]), color="C1")
    ax3.set_title("Bright unbinned")
    ax2.bar(bin_edges[:-1], np.exp(binned_hist[DARK_STATE]), np.diff(bin_edges),
            align="edge", color="C0")
    ax2.set_title("Dark binned")
    ax4.bar(bin_edges[:-1], np.exp(binned_hist[BRIGHT_STATE]), np.diff(bin_edges),
            align="edge", color="C1")
    ax4.set_title("Bright binned")
    for ax in [ax3, ax4]:
        ax.set_xlim(0, max_photons)
        ax.set_xlabel("Number of photons")
    for ax in [ax1, ax3]:
        ax.set_xlim(-0.5, max_photons)
        ax.set_ylabel("Probability of detection")
    ax2.set_xlim(0, max_photons)
    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_ylim([0, 1])
    for edge in bin_edges:
        ax2.axvline(edge, color="k", linestyle="-")
        ax4.axvline(edge, color="k", linestyle="-")
    for ax, label in zip([ax1, ax2, ax3, ax4], ["(a)", "(b)", "(c)", "(d)"]):
        ax.set_title(label, loc="left")
    fig.suptitle(r"$\Delta t = {:.2f} \mu\mathrm{{s}}$".format(
        time / dimensionful_gamma * 1e6))
    plt.tight_layout()
    filename = os.path.join(data_directory, "binned_histograms.svg")
    plt.savefig(filename)
    plt.show()


def main():
    plt.rc('text', usetex=True)

    font = {'family': 'serif', 'size': 12, 'serif': ['computer modern roman']}
    plt.rc('font', **font)
    font = {'size': 12, 'sans-serif': ['computer modern sans-serif']}

    plt.rc('font', **font)
    plt.rcParams.update({
        'text.latex.preamble': r'\usepackage{amsfonts}',
    })
    fire.Fire(plot_binned_histograms)


if __name__ == '__main__':
    main()
