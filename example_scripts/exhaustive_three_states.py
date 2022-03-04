import os

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.path.insert(0, parentdir)

import fire

import numpy as np
import torch
from perm_hmm.simulator import HMMSimulator
from perm_hmm.util import log1mexp
from perm_hmm.policies.exhaustive import ExhaustivePolicy
from perm_hmm.util import id_and_transpositions

from example_systems.three_states import three_state_hmm


def run_exhaustive_experiment(hmm, perms, steps, verbosity=0):
    exhaustive_policy = ExhaustivePolicy(perms, hmm, steps,
                                         save_history=False)
    log_value = exhaustive_policy.compute_perm_tree(return_log_costs=True,
                                                    delete_belief_tree=False,
                                                    is_cost_func=False)
    result = {
        b'log_value': log_value,
    }
    if verbosity:
        result[b"perms"] = exhaustive_policy.perm_tree
    if verbosity > 1:
        result[b"beliefs"] = exhaustive_policy.belief_tree
    return result


def run_no_perm_experiment(hmm, steps, verbosity=0):
    sim = HMMSimulator(hmm)
    retval = sim.all_classifications(steps, verbosity=verbosity)
    if verbosity == 1:
        retval[1].pop(b"posterior_log_initial_state_dist", None)
    return retval


def run_experiment(hmm, perms, steps, verbosity=0):
    exhaustive_dic = run_exhaustive_experiment(hmm, perms, steps, verbosity=verbosity)
    no_perm_result = run_no_perm_experiment(hmm, steps, verbosity=verbosity)
    return {
        b"exhaustive": exhaustive_dic,
        b"no_perms": no_perm_result,
    }


def six_step_experiment(
        verbosity=0,
):
    steps = 6
    min_log_a = -2 * np.log(10)
    min_log_b = -2 * np.log(10)
    num_grid = 10
    a_grid = np.linspace(min_log_a, 0, num_grid, endpoint=False)
    b_grid = np.linspace(min_log_b, 0, num_grid, endpoint=False)
    resultss = []
    for a in a_grid.flatten():
        results = []
        for b in b_grid.flatten():
            hmm = three_state_hmm(a, b)
            perms = id_and_transpositions(hmm.initial_logits.shape[-1])
            result = run_experiment(hmm, perms, steps, verbosity=verbosity)
            results.append({b"a": a, b"b": b, b"result": result})
        resultss.append(results)
    return resultss


def preprocess_six_step(six_step_result):
    processed = {b"a": np.array([[x[b"a"]/np.log(10) for x in y] for y in six_step_result])}
    processed[b"b"] = np.array([[x[b"b"]/np.log(10)for x in y] for y in six_step_result])
    processed[b"no_perm"] = np.array(
        [[x[b"result"][b"no_perms"].log_misclassification_rate().numpy()/np.log(10) for x in y] for y in six_step_result]
    )
    processed[b"exhaustive"] = np.array(
        [[log1mexp(x[b"result"][b"exhaustive"][b"log_value"][0][0]).numpy()/np.log(10) for x in y] for y in six_step_result]
    )
    return processed


def preprocess_two_step(two_step_result):
    processed = {b"no_perm": np.array([x[b"no_perm"]/np.log(10) for x in two_step_result])}
    processed[b"exhaustive"] = np.array([x[b"exhaustive"]/np.log(10) for x in two_step_result])
    return processed


def two_step_experiment(diagonal):
    results = []
    for a in diagonal:
        hmm = three_state_hmm(a, a)
        sim = HMMSimulator(hmm)
        no_perm_results = sim.all_classifications(2, verbosity=0)
        perms = id_and_transpositions(3)
        exhaustive_policy = ExhaustivePolicy(perms, hmm, 2, save_history=False)
        log_value = exhaustive_policy.compute_perm_tree(return_log_costs=True,
                                                        delete_belief_tree=False,
                                                        is_cost_func=False)
        result = {
            b"no_perm": no_perm_results.log_misclassification_rate().numpy(),
            b"exhaustive": log1mexp(log_value[0][0].numpy()),
        }
        results.append(result)
    return results


def plot_data(six_step_results, two_step_results):
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    a = six_step_results[b"a"]
    b = six_step_results[b"b"]
    six_exhaustive = six_step_results[b'exhaustive']
    six_no_perm = six_step_results[b'no_perm']

    two_exhaustive = two_step_results[b'exhaustive']
    two_no_perm = two_step_results[b'no_perm']

    fig = plt.figure(figsize=(8, 3))
    ax1, ax2 = fig.subplots(1, 2)
    ax1.set_aspect('equal')
    ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax1.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax2.set_xlabel('a')
    ax2.set_ylabel('b')
    ax2.set_title('No Permutation - Exhaustive')

    ax1.plot(np.diagonal(a), np.diagonal(six_no_perm), label='No Perm, $n = 6$', color=colors[0])
    ax1.plot(np.diagonal(a), two_no_perm, label="No Perm, $n = 2$", color=colors[0], linestyle='--')
    ax1.plot(np.diagonal(a), np.diagonal(six_exhaustive), label='Perm, $n = 6$', color=colors[1])
    ax1.plot(np.diagonal(a), two_exhaustive, label="Perm, $n = 2$", color=colors[1], linestyle='--')
    ax1.legend()
    ax1.set_xlabel(r'$\log_{10}(a) = \log_{10}(b)$')
    ax1.set_ylabel(r'$\log_{10}(\mathbb{P}(\hat{S_1} \neq S_1))$')
    ax1.set_title(r'Infidelities along $a = b$')
    ax1.text(-0.1, 1.15, r'(a)', transform=ax1.transAxes, fontsize=14, va='top', ha='right')

    cs = ax2.contourf(a, b, six_no_perm - six_exhaustive, cmap=plt.cm.binary, levels=8)
    ax2.plot(np.diagonal(a), np.diagonal(b), 'r--')
    ax2.set_title(r'Infidelity ratios, $n = 6$')
    ax2.set_xlabel(r'$\log_{10}(a)$')
    ax2.set_ylabel(r'$\log_{10}(b)$')
    ax2.set_xticks(np.linspace(np.min(a), np.max(a), 8))
    ax2.set_yticks(np.linspace(np.min(b), np.max(b), 8))
    ax2.set_aspect('equal')
    ax2.text(1.45, 1.15, r'(b)', transform=ax1.transAxes, fontsize=14, va='top', ha='right')
    fig.colorbar(cs, shrink=.8, label=r'$\log_{{10}}(\mathbb{{P}}_{{\mathrm{{no\;perm}}}}(\hat{{S}}_1 \neq S_1)/\mathbb{{P}}_{{\mathrm{{perm}}}}(\hat{{S}}_1 \neq S_1))$')

    return fig


def do_all_experiments(
        verbosity=0,
        data_directory=None,
):
    six_res = six_step_experiment(
        verbosity=verbosity,
    )
    if data_directory is None:
        data_directory = os.getcwd()
    filename = os.path.join(data_directory, "three_state_six_steps.pt")
    with open(filename, "wb") as f:
        torch.save(six_res, f)

    six_res = preprocess_six_step(six_res)
    diagonal = np.diagonal(six_res[b"a"]) * np.log(10)
    two_res = two_step_experiment(
        diagonal,
    )
    if data_directory is None:
        data_directory = os.getcwd()
    filename = os.path.join(data_directory, "three_state_two_steps.pt")
    with open(filename, "wb") as f:
        torch.save(two_res, f)

    two_res = preprocess_two_step(two_res)
    fig = plot_data(six_res, two_res)
    plt.figure(fig.number)
    filename = os.path.join(data_directory, "three_state_plots.svg")
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

    fire.Fire(do_all_experiments)


if __name__ == '__main__':
    main()
