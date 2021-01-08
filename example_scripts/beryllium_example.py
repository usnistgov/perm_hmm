import argparse
import torch
import pyro.distributions as dist

from perm_hmm.models.hmms import PermutedDiscreteHMM
from perm_hmm.simulator import HMMSimulator
from perm_hmm.classifiers.interrupted import IIDBinaryIntClassifier
from perm_hmm.training.interrupted_training import exact_train_binary_ic, train_binary_ic
from perm_hmm.util import num_to_data
from perm_hmm.postprocessing import ExactPostprocessor, EmpiricalPostprocessor
import perm_hmm.physical_systems.beryllium as beryllium
from perm_hmm.strategies.min_ent import MinEntropySelector
from perm_hmm.loss_functions import binary_zero_one, log_binary_zero_one

def exact_rates(phmm: PermutedDiscreteHMM, num_bins, perm_selector, classifier=None, num_ratios=20, verbosity=0):
    experiment_parameters = {
        b"hmm_params": {
            b"initial_logits": phmm.initial_logits,
            b"transition_logits": phmm.transition_logits,
            b"observation_params": phmm.observation_dist._param,
        },
        b"possible_perms": perm_selector.possible_perms,
        b"num_bins": torch.tensor(num_bins),
    }
    simulator = HMMSimulator(phmm)
    ic = IIDBinaryIntClassifier(
        dist.Bernoulli(phmm.observation_dist._param[beryllium.BRIGHT_STATE]),
        dist.Bernoulli(phmm.observation_dist._param[beryllium.DARK_STATE]),
        torch.tensor(1.),
        torch.tensor(1.),
    )
    base = len(phmm.observation_dist.enumerate_support())
    data = torch.stack(
        [num_to_data(num, num_bins, base) for num in range(base**num_bins)]
    ).float()
    lp = phmm.log_prob(data)
    plisd = phmm.posterior_log_initial_state_dist(data)
    log_joint = plisd.T + lp
    _ = exact_train_binary_ic(ic, data, log_joint, beryllium.DARK_STATE, beryllium.BRIGHT_STATE, num_ratios=num_ratios)
    nop = simulator.all_classifications(num_bins, classifier=classifier, verbosity=verbosity)
    pp = simulator.all_classifications(num_bins, classifier=classifier, perm_selector=perm_selector, verbosity=verbosity)
    if verbosity:
        nop, nod = nop
        pp, pd = pp
    i_results = ic.classify(data, verbosity=verbosity)
    if verbosity:
        i_classifications = i_results[0]
    else:
        i_classifications = i_results
    ip = ExactPostprocessor(log_joint, i_classifications)
    i_classifications = ip.classifications
    no_classifications = nop.classifications
    p_classifications = pp.classifications
    toret =  {
        b"interrupted_log_rate": ip.log_risk(log_binary_zero_one(beryllium.DARK_STATE, beryllium.BRIGHT_STATE)),
        b"permuted_log_rate": pp.log_misclassification_rate(),
        b"unpermuted_log_rate": nop.log_misclassification_rate(),
        b"interrupted_log_matrix": ip.log_confusion_matrix(),
        b"permuted_log_matrix": pp.log_confusion_matrix(),
        b"unpermuted_log_matrix": nop.log_confusion_matrix(),
        b"interrupted_classifications": i_classifications,
        b"unpermuted_classifications": no_classifications,
        b"permuted_classifications": p_classifications,
        b"experiment_parameters": experiment_parameters
    }
    if verbosity:
        toret[b"unpermuted_extras"] = nod
        toret[b"permuted_extras"] = pd
        toret[b"interrupted_break_ratio"] = i_results[1:]
    return toret

def empirical_rates(phmm: PermutedDiscreteHMM, num_bins, perm_selector, classifier=None, num_ratios=20, num_train=1000, num_samples=1000, confidence=.95, verbosity=0):
    experiment_parameters = {
        b"hmm_params": {
            b"initial_logits": phmm.initial_logits,
            b"transition_logits": phmm.transition_logits,
            b"observation_params": phmm.observation_dist._param,
        },
        b"possible_perms": perm_selector.possible_perms,
        b"num_bins": torch.tensor(num_bins),
    }
    simulator = HMMSimulator(phmm)
    ic = IIDBinaryIntClassifier(
        dist.Bernoulli(phmm.observation_dist._param[beryllium.BRIGHT_STATE]),
        dist.Bernoulli(phmm.observation_dist._param[beryllium.DARK_STATE]),
        torch.tensor(1.),
        torch.tensor(1.),
    )
    x, training_data = phmm.sample((num_train, num_bins))
    _ = train_binary_ic(ic, training_data, x[..., 0], beryllium.DARK_STATE, beryllium.BRIGHT_STATE, num_ratios=num_ratios)
    pp = simulator.simulate(num_bins, num_samples, classifier=classifier, perm_selector=perm_selector, verbosity=verbosity)
    nop, d = simulator.simulate(num_bins, num_samples, classifier=classifier, verbosity=max(1, verbosity))
    if verbosity:
        pp, pd = pp
    i_results = ic.classify(d[b"data"], verbosity=verbosity)
    if verbosity:
        i_classifications, i_dict = i_results
    else:
        i_classifications = i_results

    # HACK: We need to translate the binary classifications to the HMM classifications.
    # For purposes of computing misclassification rates, we can instead use the
    # binary_zero_one loss function, but I haven't implemented a confidence interval
    # for that, so we just do this instead.
    testing_states = torch.tensor([beryllium.DARK_STATE, beryllium.BRIGHT_STATE])
    ip = EmpiricalPostprocessor(nop.ground_truth, testing_states[i_classifications.long()])

    i_classifications = ip.classifications
    no_classifications = nop.classifications
    p_classifications = pp.classifications
    toret = {
        b"interrupted_rates": ip.misclassification_rate(confidence),
        b"permuted_rates": pp.misclassification_rate(confidence),
        b"unpermuted_rates": nop.misclassification_rate(confidence),
        b"interrupted_classifications": i_classifications,
        b"unpermuted_classifications": no_classifications,
        b"permuted_classifications": p_classifications,
        b"experiment_parameters": experiment_parameters
    }
    if verbosity:
        toret[b"unpermuted_extras"] = d
        toret[b"permuted_extras"] = pd
        toret[b"interrupted_extras"] = i_dict
        toret[b"training_states"] = x
        toret[b"training_data"] = training_data
    return toret

def main(args):
    if args.integration_time:
        integration_time = args.integration_time
    elif args.total_time:
        integration_time = args.total_time / args.num_bins
    else:
        integration_time = args.total_time_over_1e7*1e-7 / args.num_bins
    bright_or_dark, pij, bright_probs = \
        [torch.from_numpy(x).float()
         for x in beryllium.bernoulli_parameters(integration_time)]
    output_dist = dist.Bernoulli(bright_probs)
    perm_hmm = PermutedDiscreteHMM(bright_or_dark, pij.log(), output_dist)
    num_bins = args.num_bins
    perm_selector = MinEntropySelector(
        torch.from_numpy(beryllium.allowable_permutations()), perm_hmm)
    print("Running simulation, please wait...")
    if args.save_raw_data:
        verbosity = 2
    else:
        verbosity = 0
    if args.exact:
        d = exact_rates(perm_hmm, num_bins, perm_selector, verbosity=verbosity)
    elif args.approximate:
        d = empirical_rates(perm_hmm, num_bins, perm_selector, num_train=args.num_training_samples, num_samples=args.num_samples, verbosity=verbosity)
    else:
        d = {}
    print("Done.\n")
    d[b"args"] = args
    print("Writing to file...")
    filename = args.filename
    if not args.filename.split(".")[-1] == "pt":
        filename += ".pt"
    with open(filename, 'wb') as f:
        torch.save(d, f)
    print("Done.\n")



if __name__ == "__main__":
    # TODO: Add postselection
    parser = argparse.ArgumentParser(
        "An example of either the exact or approximate misclassification rate computation from Beryllium"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-i",
        "--integration-time",
        help="The amount of time to integrate the collection of photons for."
             "Pass something that can be cast as a float, e.g. 1.1e-07",
        type=float,
    )
    group.add_argument(
        "-t",
        "--total-time",
        help="Total amount of time to collect data per run. "
             "Pass something that can be cast as a float, e.g. 3.3e-05",
        type=float
    )
    group.add_argument(
        "-7",
        "--total-time-over-1e7",
        help="Total amount of time to collect data per run, in units of 1e-7 s",
        type=float
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "-e",
        "--exact",
        help="If indicated, will run an exact misclassification rate computation.",
        action="store_true",
    )
    group.add_argument(
        "-a",
        "--approximate",
        help="If indicated, will run an approximate misclassification rate computation."
             " If specified, must also specify number of samples with -s and "
             "number of training samples to train the interrupted classifier with"
             " with -t.",
        action="store_true",
    )
    parser.add_argument(
        "-s",
        "--num-samples",
        help="Number of samples to sample from the various models, "
             "and evaluate the classifiers with. Specify if and only if "
             "-a is specified.",
        type=int,
    )
    parser.add_argument(
        "-r",
        "--num-training-samples",
        help="Number of samples to sample from the classifier without "
             "permutations to train the interrupted classifier with. Specify "
             "if and only if -a is specified.",
        type=int,
    )
    parser.add_argument(
        "-o",
        "--filename",
        help=".pt filename to write to. Extension will be added if not already .pt,"
             " extension will be appended.",
        type=str,
    )
    parser.add_argument(
        "-w",
        "--save-raw-data",
        help="If indicated, will save all the data generated.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-p",
        "--prob-to-keep",
        help="The percentage of data to keep in postselection. If not "
             "specified, postselection is not performed.",
        type=float,
    )
    parser.add_argument(
        "num_bins",
        metavar="num-bins",
        help="The number discrete time steps to take data for per run.",
        type=int,
    )
    args = parser.parse_args()
    if args.approximate and (not args.num_samples or not args.num_training_samples):
        parser.error(
            "If approximate simulation is specified, both number of samples and"
            " number of training samples must be specified.")
    main(args)
