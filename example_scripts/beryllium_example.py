"""
An example using the parameters from a Beryllium 9 system.
"""
import argparse
import torch
import pyro.distributions as dist
from bayes_perm_hmm.min_entropy_hmm import PermutedDiscreteHMM
from bayes_perm_hmm.simulator import Simulator
from bayes_perm_hmm.physical_systems import beryllium


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
    perm_hmm = PermutedDiscreteHMM(bright_or_dark, pij.log(), output_dist,
                                   torch.from_numpy(
                                       beryllium.allowable_permutations()))
    num_bins = args.num_bins
    b_sim = Simulator(
        perm_hmm,
        torch.tensor([beryllium.BRIGHT_STATE, beryllium.DARK_STATE]),
        num_bins,
    )
    print("Running simulation, please wait...")
    if args.exact:
        td = b_sim.exact_train_ic()
        results = b_sim.exact_simulation()
    elif args.approximate:
        td = b_sim.train_ic(args.num_training_samples)
        results = b_sim.empirical_simulation(args.num_samples)
    ip = results.interrupted_postprocessor
    np = results.naive_postprocessor
    bp = results.bayes_postprocessor
    print("Done.\n")
    if args.prob_to_keep:
        if args.approximate:
            i_mask = ip.postselection_percentage_mask(args.prob_to_keep)
            ip = ip.postselect(i_mask)
            n_mask = np.postselection_percentage_mask(args.prob_to_keep)
            np = np.postselect(n_mask)
            b_mask = bp.postselection_percentage_mask(args.prob_to_keep)
            bp = bp.postselect(b_mask)
            i_rates = ip.misclassification_rates()
            n_rates = np.misclassification_rates()
            b_rates = bp.misclassification_rates()
        elif args.exact:
            i_rates = ip.postselected_misclassification(args.prob_to_keep)
            n_rates = np.postselected_misclassification(args.prob_to_keep)
            b_rates = bp.postselected_misclassification(args.prob_to_keep)
    else:
        i_rates = ip.misclassification_rates()
        n_rates = np.misclassification_rates()
        b_rates = bp.misclassification_rates()
    if args.approximate:
        print("False positive rate in the interrupted scheme: {}".format(i_rates.confusions.rate[beryllium.DARK_STATE, beryllium.BRIGHT_STATE]))
        print("False negative rate in the interrupted scheme: {}".format(i_rates.confusions.rate[beryllium.BRIGHT_STATE, beryllium.DARK_STATE]))
        print("Average misclassification rate in the interrupted scheme: {}".format(i_rates.average.rate))
        print("\n")
        print("False positive rate in the naive scheme: {}".format(n_rates.confusions.rate[beryllium.DARK_STATE, beryllium.BRIGHT_STATE]))
        print("False negative rate in the naive scheme: {}".format(n_rates.confusions.rate[beryllium.BRIGHT_STATE, beryllium.DARK_STATE]))
        print("Average misclassification rate in the naive scheme: {}".format(n_rates.average.rate))
        print("\n")
        print("False positive rate in the permuted scheme: {}".format(b_rates.confusions.rate[beryllium.DARK_STATE, beryllium.BRIGHT_STATE]))
        print("False negative rate in the permuted scheme: {}".format(b_rates.confusions.rate[beryllium.BRIGHT_STATE, beryllium.DARK_STATE]))
        print("Average misclassification rate in the permuted scheme: {}".format(b_rates.average.rate))
        print("\n")
    else:
        print("False positive rate in the interrupted scheme: {}".format(i_rates.confusions[beryllium.DARK_STATE, beryllium.BRIGHT_STATE]))
        print("False negative rate in the interrupted scheme: {}".format(i_rates.confusions[beryllium.BRIGHT_STATE, beryllium.DARK_STATE]))
        print("Average misclassification rate in the interrupted scheme: {}".format(i_rates.average))
        print("\n")
        print("False positive rate in the naive scheme: {}".format(n_rates.confusions[beryllium.DARK_STATE, beryllium.BRIGHT_STATE]))
        print("False negative rate in the naive scheme: {}".format(n_rates.confusions[beryllium.BRIGHT_STATE, beryllium.DARK_STATE]))
        print("Average misclassification rate in the naive scheme: {}".format(n_rates.average))
        print("\n")
        print("False positive rate in the permuted scheme: {}".format(b_rates.confusions[beryllium.DARK_STATE, beryllium.BRIGHT_STATE]))
        print("False negative rate in the permuted scheme: {}".format(b_rates.confusions[beryllium.BRIGHT_STATE, beryllium.DARK_STATE]))
        print("Average misclassification rate in the permuted scheme: {}".format(b_rates.average))
        print("\n")

    i_classifications = ip.classifications
    n_classifications = np.classifications
    b_classifications = bp.classifications

    if args.filename:
        print("Writing to file...")
        filename = args.filename
        if not args.filename.split(".")[-1] == "pt":
            filename += ".pt"
        to_save = {
            b"experiment_parameters": b_sim.experiment_parameters,
            b"interrupted_rates": i_rates,
            b"naive_rates": n_rates,
            b"bayes_rates": b_rates,
            b"interrupted_classifications": i_classifications,
            b"naive_classifications": n_classifications,
            b"bayes_classifications": b_classifications,
            b"args": args,
        }
        if args.save_raw_data:
            to_save[b"runs"] = results.runs
            if args.approximate:
                to_save[b"training_runs"] = td
        with open(filename, 'wb') as f:
             torch.save(
                to_save,
                f,
            )
        print("Done.\n")


if __name__ == "__main__":
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
