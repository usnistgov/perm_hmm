import argparse
import torch
import pyro.distributions as dist

from perm_hmm.models.hmms import PermutedDiscreteHMM
import perm_hmm.example_systems.beryllium as beryllium
from perm_hmm.strategies.min_ent import MinEntropySelector
from perm_hmm.rate_comparisons import exact_binary_rates, empirical_binary_rates


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
        d = exact_binary_rates(perm_hmm, num_bins, perm_selector, beryllium.DARK_STATE, beryllium.BRIGHT_STATE, verbosity=verbosity)
    elif args.approximate:
        d = empirical_binary_rates(perm_hmm, num_bins, perm_selector, beryllium.DARK_STATE, beryllium.BRIGHT_STATE, num_train=args.num_training_samples, num_samples=args.num_samples, verbosity=verbosity)
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
