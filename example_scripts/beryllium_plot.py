import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parentdir)
parentdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parentdir)
import fire

import numpy as np
import matplotlib.pyplot as plt

import torch
import pyro.distributions as dist

from adapt_hypo_test.two_states.util import log1mexp
from example_systems.beryllium import DARK_STATE, BRIGHT_STATE, N_STATES, dimensionful_gamma, expanded_permutations, simplest_perms, expanded_initial, expanded_outcomes, expanded_transitions, log_prob_n_given_l, unbinned_hists
from perm_hmm.simulator import HMMSimulator
from perm_hmm.util import log_tvd
from perm_hmm.models.hmms import ExpandedHMM
from perm_hmm.policies.exhaustive import SplitExhaustivePolicy, ExhaustivePolicy
from perm_hmm.policies.min_tree import MinEntPolicy
from perm_hmm.policies.belief import HMMBeliefState
from perm_hmm.simulator import HMMSimulator
from perm_hmm.binning import optimally_binned_consecutive, binned_expanded_hmm



def run_exhaustive_experiment(hmm, perms, steps, verbosity=0):
    root_belief = HMMBeliefState.from_expandedhmm(hmm)
    exhaustive_policy = ExhaustivePolicy(perms, hmm, steps, root_belief=root_belief, save_history=False)
    log_value = exhaustive_policy.compute_perm_tree(return_log_costs=True, delete_belief_tree=False, is_cost_func=False)
    result = {
        b'log_value': log_value,
    }
    if verbosity:
        result[b"perms"] = exhaustive_policy.perm_tree
    if verbosity > 1:
        result[b"beliefs"] = exhaustive_policy.belief_tree
    return result


def run_min_ent_experiment(hmm, perms, steps, verbosity=0):
    sim = HMMSimulator(hmm)
    root_belief = HMMBeliefState.from_expandedhmm(hmm)
    policy = MinEntPolicy(perms, hmm, look_ahead=2, root_belief=root_belief)
    retval = sim.all_classifications(steps, perm_policy=policy, verbosity=verbosity)
    if verbosity == 1:
        retval[1].pop(b"history", None)
        retval[1].pop(b"posterior_log_initial_state_dist", None)
    return retval


def run_no_perm_experiment(hmm, steps, verbosity=0):
    sim = HMMSimulator(hmm)
    retval = sim.all_classifications(steps, verbosity=verbosity)
    if verbosity == 1:
        retval[1].pop(b"posterior_log_initial_state_dist", None)
    return retval


def run_histogram_method(time):
    dark_bright_histograms = unbinned_hists(time, max_photons=50)
    return log1mexp(np.logaddexp(0, log_tvd(dark_bright_histograms[0], dark_bright_histograms[1])) - np.log(2))


def run_experiment(hmm, perms, steps, verbosity=0):
    exhaustive_dic = run_exhaustive_experiment(hmm, perms, steps, verbosity=verbosity)
    min_ent_result = run_min_ent_experiment(hmm, perms, steps, verbosity=verbosity)
    no_perm_result = run_no_perm_experiment(hmm, steps, verbosity=verbosity)
    return {
        b"exhaustive": exhaustive_dic,
        b"min_entropy": min_ent_result,
        b"no_perms": no_perm_result,
    }



def make_filename(path, params):
    assert os.path.exists(path)
    fn = os.path.join(path, "time_{:.3e}_steps_{}.pt".format(params[b"dimensionful_time"],
                                                             params[b"steps"]))
    return fn



def setup_experiment(time, steps, max_photons=15, num_bins=4):
    r"""Given an integration time, bins the output distributions such that the
    symmetrized divergence from the bright state to the dark state is maximal,
    then returns everything used to compute that, along with the resulting HMM.

    :param time:
    :return:
    """
    initial_logits = expanded_initial(max_photons)
    transition_logits = expanded_transitions(time, max_photons)
    observation_dist = dist.Categorical(logits=torch.from_numpy(expanded_outcomes(max_photons)))
    expanded_hmm = ExpandedHMM(
        torch.from_numpy(initial_logits),
        torch.from_numpy(transition_logits),
        observation_dist,
    )
    edges, min_infidelity = optimally_binned_consecutive(expanded_hmm, num_bins, steps=steps)
    hmm = binned_expanded_hmm(expanded_hmm, edges)
    retval = {b"hmm": hmm, b"time": time,
              b"dimensionful_time": time / dimensionful_gamma,
              b"max_photons": max_photons, b"num_bins": num_bins,
              b"steps": steps, b"histogram_result": run_histogram_method(time*steps)}
    return retval


def do_experiment(num_bins=3,
                  steps=2,
                  max_dim_tot_time=3.3e-4,
                  num_time_points=4,
                  verbosity=0):
    max_tot_time = max_dim_tot_time * dimensionful_gamma
    max_integ_time = max_tot_time / steps
    inc = max_integ_time / num_time_points
    time_range = np.linspace(inc, max_integ_time, num_time_points, endpoint=True)
    results = []
    for time in time_range:
        sub_params = setup_experiment(time, steps, num_bins=num_bins)
        sub_params[b"perms"] = torch.from_numpy(expanded_permutations(simplest_perms(), k=num_bins))
        hmm = sub_params[b"hmm"]
        perms = sub_params[b"perms"]
        steps = sub_params[b"steps"]
        data = run_experiment(hmm, perms, steps, verbosity=verbosity)
        result = {
            b"params": sub_params,
            b"result": data,
        }
        result[b"result"][b"histogram"] = sub_params[b"histogram_result"]
        results.append(result)
    return results


def preprocess_results(results):
    steps = results[0][b"params"][b"steps"]
    bins = results[0][b"params"][b"num_bins"]
    dim_times = np.array([r[b"params"][b"dimensionful_time"] for r in results])
    no_perm_rates = np.array([r[b"result"][b"no_perms"].log_misclassification_rate().numpy()/np.log(10) for r in results])
    min_ent_rates = np.array([r[b"result"][b"min_entropy"].log_misclassification_rate().numpy()/np.log(10) for r in results])
    exhaustive_rates = np.array([log1mexp(r[b"result"][b"exhaustive"][b"log_value"][0][0])/np.log(10) for r in results])
    histogram_rates = np.array([r[b"result"][b"histogram"]/np.log(10) for r in results])
    return {
        b"steps": steps,
        b"bins": bins,
        b"dim_times": dim_times,
        b"no_perm_rates": no_perm_rates,
        b"min_ent_rates": min_ent_rates,
        b"exhaustive_rates": exhaustive_rates,
        b"histogram_rates": histogram_rates,
    }


def plot_results(results):
    dim_times = results[b"dim_times"]
    no_perm_rates = results[b"no_perm_rates"]
    min_ent_rates = results[b"min_ent_rates"]
    exhaustive_rates = results[b"exhaustive_rates"]
    histogram_rates = results[b"histogram_rates"]
    steps = results[b"steps"]
    bins = results[b"bins"]

    dim_times = dim_times * 1e6
    # Plot the results
    fig = plt.figure(figsize=(8.5, 4))
    axs = fig.subplots(1, 3)
    for i, label in enumerate(["(a)", "(b)", "(c)"]):
        time_range = dim_times[
                     i * len(dim_times) // 3:(i + 1) * len(dim_times) // 3]
        ax = axs[i]
        ax.plot(time_range, exhaustive_rates[
                            i * len(dim_times) // 3:(i + 1) * len(
                                dim_times) // 3], label='Exhaustive')
        ax.plot(time_range, min_ent_rates[
                            i * len(dim_times) // 3:(i + 1) * len(
                                dim_times) // 3], dashes=[1, 1],
                label='Min Entropy')
        ax.plot(time_range, no_perm_rates[
                            i * len(dim_times) // 3:(i + 1) * len(
                                dim_times) // 3], linestyle="--",
                label='No Perms')
        ax.plot(time_range, histogram_rates[
                            i * len(dim_times) // 3:(i + 1) * len(
                                dim_times) // 3], linestyle="-.",
                label='Histogram')
        ax.set_xlabel(r"Total time ($\mu$s)")
        ax.set_title(label, loc="left")
        if i == 0:
            ax.set_ylabel(r"$\log_{{10}}(\mathbb{{P}}(\hat{{L}}_1 \neq L_1))$")
            ax.legend()
    fig.suptitle("{} Bins, {} Steps".format(bins, steps))
    plt.tight_layout()
    return fig


def make_and_plot_data(
        num_bins=3,
        steps=2,
        max_dim_tot_time=3.3e-4,
        num_time_points=9,
        verbosity=0,
        data_directory=None,
):
    results = do_experiment(num_bins=num_bins,
                            steps=steps,
                            max_dim_tot_time=max_dim_tot_time,
                            num_time_points=num_time_points,
                            verbosity=verbosity)
    if data_directory is None:
        data_directory = os.getcwd()
    filename = os.path.join(data_directory, "bins_{}_steps_{}.pt".format(num_bins, steps))
    with open(filename, "wb") as f:
        torch.save(results, f)
    results = preprocess_results(results)
    fig = plot_results(results)
    plt.figure(fig.number)
    plt.savefig(os.path.join(data_directory, "bins_{}_steps_{}.svg".format(num_bins, steps)))
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
    fire.Fire(make_and_plot_data)


if __name__ == '__main__':
    main()