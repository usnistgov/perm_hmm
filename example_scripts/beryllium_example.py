import argparse
import torch
import pyro.distributions as dist

from perm_hmm.models.hmms import PermutedDiscreteHMM
from perm_hmm.simulator import HMMSimulator
from perm_hmm.classifiers.interrupted import IIDInterruptedClassifier
from perm_hmm.training.interrupted_training import exact_train_ic, train_ic
from perm_hmm.util import num_to_data
# from perm_hmm.postprocessing.interrupted_postprocessors import InterruptedExactPostprocessor, InterruptedEmpiricalPostprocessor
from perm_hmm.postprocessing import ExactPostprocessor, EmpiricalPostprocessor
import perm_hmm.physical_systems.beryllium as beryllium
from perm_hmm.strategies.min_ent import MinEntropySelector


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
    ic = IIDInterruptedClassifier(
        phmm.observation_dist,
        torch.tensor(1.),
    )
    base = len(phmm.observation_dist.enumerate_support())
    data = torch.stack(
        [num_to_data(num, num_bins, base) for num in range(base**num_bins)]
    ).float()
    lp = phmm.log_prob(data)
    plisd = phmm.posterior_log_initial_state_dist(data)
    log_joint = plisd.T + lp
    _ = exact_train_ic(ic, data, log_joint, num_ratios=num_ratios)
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
    ip = ExactPostprocessor(nop.log_joint, i_classifications)
    i_classifications = ip.classifications
    no_classifications = nop.classifications
    p_classifications = pp.classifications
    toret =  {
        b"interrupted_log_rate": ip.log_misclassification_rate(),
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
    ic = IIDInterruptedClassifier(
        phmm.observation_dist,
        torch.tensor(1.),
    )
    x, training_data = phmm.sample((num_train, num_bins))
    _ = train_ic(ic, training_data, x[..., 0], num_ratios=num_ratios)
    pp = simulator.simulate(num_bins, num_samples, classifier=classifier, perm_selector=perm_selector, verbosity=verbosity)
    nop, d = simulator.simulate(num_bins, num_samples, classifier=classifier, verbosity=max(1, verbosity))
    if verbosity:
        pp, pd = pp
    i_results = ic.classify(d[b"data"], verbosity=verbosity)
    if verbosity:
        i_classifications, i_dict = i_results
    else:
        i_classifications = i_results
    ip = EmpiricalPostprocessor(nop.ground_truth, i_classifications)
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

#
#
#
#
#
# class Simulator(object):
#     """
#     Simulates an experiment with Bernoulli outcomes.
#
#     Simulates both an experiment where all possible runs are computed
#     (by brute force, in exponential time), or an approximate experiment,
#     where we sample from the HMMs and estimate the resulting misclassification
#     rates. By default, the experiments include comparisons to both a naive HMM
#     where no permutations occur and we use a MAP estimator, and an "interrupted"
#     classifier, which terminates a data collection cycle if a likelihood
#     threshold has been crossed.
#     """
#
#     def __init__(self, phmm, testing_states, num_bins):
#         """
#         Initializes the experiment.
#
#         :param bayes_perm_hmm.hmms.PermutedDiscreteHMM phmm:
#             the model whose
#             misclassification rate will be computed. The naive_hmm parameters
#             will be classified from those of bayes_hmm.
#         :param torch.Tensor testing_states: states to perform hypothesis tests for.
#         :param int num_bins: time dimension of data to be collected.
#
#         :raises: ValueError
#         """
#         self.hmm = phmm
#         """:py:class:`PermutedDiscreteHMM`
#         The model whose misclassification rates we wish to analyze.
#         """
#         num_states = len(self.hmm.initial_logits)
#         lts = testing_states.tolist()
#         sts = set(lts)
#         if not len(sts) == len(lts):
#             raise ValueError("States to test for must be unique.")
#         if len(lts) <= 1:
#             raise ValueError("Must attempt to discriminate between at least two"
#                              "states.")
#         if not set(testing_states.tolist()).issubset(range(num_states)):
#             raise ValueError("States to test for must be states of the model.")
#         self.testing_states = testing_states
#         """
#         states to test for.
#         """
#         self.num_bins = num_bins
#         """:py:class:`int`
#         The number of time steps to compute for.
#         """
#         self.ic = InterruptedClassifier(
#             self.hmm.observation_dist,
#             testing_states,
#         )
#         """:py:class:`InterruptedClassifier`
#         The corresponding model to the input `bayes_hmm`
#         whose performance we wish to compare to
#         """
#         # self.experiment_parameters = ExperimentParameters(
#         #     PermutedParameters(
#         #         self.hmm.initial_logits,
#         #         self.hmm.transition_logits,
#         #         self.hmm.observation_dist._param,
#         #         self.hmm.possible_perms,
#         #     ),
#         #     self.testing_states,
#         #     torch.tensor(self.num_bins),
#         # )
#
#     def exact_train_ic(self, num_ratios=20):
#         base = len(self.hmm.observation_dist.enumerate_support())
#         data = torch.stack(
#             [num_to_data(num, self.num_bins, base) for num in range(base**self.num_bins)]
#         ).float()
#         naive_post_dist = self.hmm.posterior_log_initial_state_dist(data)
#         naive_lp = self.hmm.log_prob(data)
#         _ = exact_train_ic(self.ic, data, naive_lp, naive_post_dist, self.hmm.initial_logits, num_ratios=num_ratios)
#         return naive_lp, naive_post_dist
#
#     def train_ic(self, num_train, num_ratios=20):
#         """
#         Trains the :py:class:`InterruptedClassifier` with `num_train` number of
#         training runs, to find the optimal threshold likelihood ratios.
#         Returns the training data, having already passed it to `self.ic.train`,
#         so the training data is returned just in case we want to record it.
#
#         :param int num_train: The number of runs to train the
#             :py:class:`InterruptedClassifier` with.
#         :returns: A :py:class:`HMMOutput` object containing
#
#             .states: :py:class:`torch.Tensor` int
#             The actual states which generated the data in `.observations`.
#
#                 shape ``(num_train, num_bins)``
#
#             .observations: :py:class:`torch.Tensor` float.
#             The data which was used to train the classifier.
#
#                 shape ``(num_train, num_bins)``
#
#         .. seealso:: method
#             :py:meth:`InterruptedClassifier.train`
#         """
#         hmm_out = self.hmm.sample((num_train, self.num_bins))
#         states = hmm_out.states
#         observations = hmm_out.observations
#         ground_truth = states[:, 0]
#         _ = train_ic(self.ic, observations, ground_truth, self.hmm.initial_logits.shape[-1], num_ratios=num_ratios)
#         return hmm_out
#
#     def exact_simulation(self, possible_perms, return_raw=True):
#         """
#         Computes the data required to compute the exact misclassification rates
#         of the three models under consideration, the HMM with permtuations,
#         the HMM without permtuations, and the InterruptedClassifier.
#
#         :returns: :py:class:`ExactResults` with components
#
#             .interrupted_postprocessor: :py:class:`InterruptedExactPostprocessor`.
#             Use it to compute the
#             misclassification rates of the :py:class:`InterruptedClassifier`.
#
#             .naive_postprocessor: :py:class:`PostDistExactPostprocessor`.
#             Use it to compute the misclassification rates of the HMM classifier
#             without permutations.
#
#             .bayes_postprocessor: :py:class:`PostDistExactPostprocessor`.
#             Use it to compute the misclassification rates of the HMM classifier
#             with permutations.
#
#             .runs: :py:class:`ExactRun`
#             All the data which was used to produce the objects.
#             returned just in case we would like to save it for later. See
#             :py:meth:`BernoulliSimulator._exact_single_run`
#             for details on members.
#         """
#         if self.ic.ratio is None:
#             raise ValueError("Must train interrupted classifier first. Use .exact_train_ic")
#         base = len(self.hmm.observation_dist.enumerate_support())
#         data = torch.stack(
#             [num_to_data(num, self.num_bins, base) for num in range(base**self.num_bins)]
#         ).float()
#         naive_lp = self.hmm.log_prob(data)
#         naive_dist = self.hmm.posterior_log_initial_state_dist(data)
#         bayes_result = self.hmm.get_perms(data, save_history=return_raw)
#         if return_raw:
#             perm = bayes_result.perm
#         else:
#             perm = bayes_result
#         bayes_lp = self.hmm.log_prob_with_perm(data, perm)
#         bayes_plisd = \
#             self.hmm.posterior_log_initial_state_dist(data, perm)
#         interrupted_results = self.ic.classify(data)
#         if return_raw:
#             results = ExactRun(
#                 data,
#                 bayes_result,
#             )
#         naive_ep = PostDistExactPostprocessor(
#             naive_lp,
#             naive_dist,
#             self.hmm.initial_logits,
#             self.testing_states,
#         )
#         bayes_ep = PostDistExactPostprocessor(
#             bayes_lp,
#             bayes_plisd,
#             self.hmm.initial_logits,
#             self.testing_states,
#         )
#         ip = InterruptedExactPostprocessor(
#             naive_lp,
#             naive_dist,
#             self.hmm.initial_logits,
#             self.testing_states,
#             interrupted_results,
#         )
#         if return_raw:
#             return ExactResults(
#                 ip,
#                 naive_ep,
#                 bayes_ep,
#                 results,
#             )
#         else:
#             return ExactNoResults(
#                 ip,
#                 naive_ep,
#                 bayes_ep,
#             )
#
#     def empirical_simulation(self, num_samples, return_raw=True):
#         """
#         Computes the data required to compute the empirical misclassification
#         rates of the three models under consideration, the HMM with
#         permutations, the HMM without permutations, and the
#         InterruptedClassifier.
#
#         :returns: A :py:class:`EmpiricalResults` containing:
#
#             .interrupted_postprocessor: :py:class:`InterruptedEmpiricalPostprocessor`.
#             Use it to compute the
#             misclassification rates of the :py:class:`InterruptedClassifier`.
#
#             .naive_postprocessor: :py:class:`PostDistEmpiricalPostprocessor`.
#             Use it to compute the misclassification rates of the HMM classifier
#             without permutations.
#
#             .bayes_postprocessor: :py:class:`PostDistEmpiricalPostprocessor`.
#             Use it to compute the misclassification rates of the HMM classifier
#             with permutations.
#
#             .runs: :py:class:`EmpiricalRun`
#             All the data which was used to produce the objects.
#             returned just in case we would like to save it for later. See
#             :py:meth:`BernoulliSimulator._single_sampled_simulation`
#             for details on members.
#
#         .. seealso:: method
#             :py:meth:`BernoulliSimulator._single_sampled_simulation`
#         """
#         if self.ic.ratio is None:
#             raise ValueError("Must train interrupted classifier first with .train_ic")
#         bayes_output = \
#             self.hmm.sample_min_entropy((num_samples, self.num_bins))
#         naive_output = self.hmm.sample((num_samples, self.num_bins))
#         naive_data = naive_output.observations
#         naive_plisd = self.hmm.posterior_log_initial_state_dist(naive_data)
#         interrupted_results = self.ic.classify(naive_data)
#         results = EmpiricalRun(
#             HMMOutPostDist(
#                 naive_output,
#                 naive_plisd,
#             ),
#             bayes_output,
#         )
#         naive_ep = PostDistEmpiricalPostprocessor(
#             results.naive.hmm_output.states[..., 0],
#             self.testing_states,
#             self.hmm.initial_logits.shape[-1],
#             results.naive.log_post_dist,
#         )
#         bayes_ep = PostDistEmpiricalPostprocessor(
#             results.bayes.states[..., 0],
#             self.testing_states,
#             self.hmm.initial_logits.shape[-1],
#             results.bayes.history.partial_post_log_init_dists[..., -1, :],
#         )
#         ip = InterruptedEmpiricalPostprocessor(
#             results.naive.hmm_output.states[..., 0],
#             self.testing_states,
#             self.hmm.initial_logits.shape[-1],
#             *interrupted_results
#         )
#         if return_raw:
#             return EmpiricalResults(
#                 ip,
#                 naive_ep,
#                 bayes_ep,
#                 results,
#             )
#         else:
#             return EmpiricalNoResults(
#                 ip,
#                 naive_ep,
#                 bayes_ep,
#             )
