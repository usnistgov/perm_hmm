"""
Simulates the initial state discrimination experiment using different
methods, to compare the resulting error rates.
"""
import warnings
from typing import NamedTuple

import torch

from bayes_perm_hmm.interrupted import InterruptedClassifier
from bayes_perm_hmm.hmms import SampleableDiscreteHMM
from bayes_perm_hmm.return_types import ExperimentParameters, ExactRun, \
    HMMOutPostDist, EmpiricalRun, PermutedParameters
from bayes_perm_hmm.util import num_to_data
from bayes_perm_hmm.training import train, exact_train
from bayes_perm_hmm.postprocessing import InterruptedExactPostprocessor, \
    PostDistExactPostprocessor, InterruptedEmpiricalPostprocessor, \
    PostDistEmpiricalPostprocessor

exact_fields = [
    ('interrupted_postprocessor', InterruptedExactPostprocessor),
    ('naive_postprocessor', PostDistExactPostprocessor),
    ('bayes_postprocessor', PostDistExactPostprocessor),
]

ExactNoResults = NamedTuple('ExactNoResults', exact_fields)
ExactResults = NamedTuple(
    'ExactResults',
    exact_fields + [('runs', ExactRun)],
)

empirical_fields = [
    ('interrupted_postprocessor', InterruptedEmpiricalPostprocessor),
    ('naive_postprocessor', PostDistEmpiricalPostprocessor),
    ('bayes_postprocessor', PostDistEmpiricalPostprocessor),
]

EmpiricalNoResults = NamedTuple('EmpiricalNoResults', empirical_fields)
EmpiricalResults = NamedTuple(
    'EmpiricalResults',
    empirical_fields + [('runs', EmpiricalRun)],
)


class Simulator(object):
    """
    Simulates an experiment with Bernoulli outcomes.

    Simulates both an experiment where all possible runs are computed
    (by brute force, in exponential time), or an approximate experiment,
    where we sample from the HMMs and estimate the resulting misclassification
    rates. By default, the experiments include comparisons to both a naive HMM
    where no permutations occur and we use a MAP estimator, and an "interrupted"
    classifier, which terminates a data collection cycle if a likelihood
    threshold has been crossed.
    """

    def __init__(self, bayes_hmm, testing_states, num_bins):
        """
        Initializes the experiment.

        :param bayes_perm_hmm.min_entropy.PermutedDiscreteHMM bayes_hmm:
            the model whose
            misclassification rate will be computed. The naive_hmm parameters
            will be classified from those of bayes_hmm.
        :param torch.Tensor testing_states: states to perform hypothesis tests for.
        :param int num_bins: time dimension of data to be collected.

        :raises: ValueError
        """
        self.bayes_hmm = bayes_hmm
        """:py:class:`PermutedDiscreteHMM`
        The model whose misclassification rates we wish to analyze.
        """
        num_states = len(self.bayes_hmm.initial_logits)
        lts = testing_states.tolist()
        sts = set(lts)
        if not len(sts) == len(lts):
            raise ValueError("States to test for must be unique.")
        if len(lts) <= 1:
            raise ValueError("Must attempt to discriminate between at least two"
                             "states.")
        if not set(testing_states.tolist()).issubset(range(num_states)):
            raise ValueError("States to test for must be states of the model.")
        self.testing_states = testing_states
        """
        states to test for.
        """
        self.num_bins = num_bins
        """:py:class:`int`
        The number of time steps to compute for.
        """
        self.hmm = SampleableDiscreteHMM(self.bayes_hmm.initial_logits,
                                         self.bayes_hmm.transition_logits,
                                         self.bayes_hmm.observation_dist)
        """:py:class:`SampleableDiscreteHMM`
        The corresponding model the input `bayes_hmm` whose performance we wish
        to compare to.
        """
        self.ic = InterruptedClassifier(
            self.bayes_hmm.observation_dist,
            testing_states,
        )
        """:py:class:`InterruptedClassifier`
        The corresponding model to the input `bayes_hmm`
        whose performance we wish to compare to
        """
        self.experiment_parameters = ExperimentParameters(
            PermutedParameters(
                self.bayes_hmm.initial_logits,
                self.bayes_hmm.transition_logits,
                self.bayes_hmm.observation_dist._param,
                self.bayes_hmm.possible_perms,
            ),
            self.testing_states,
            torch.tensor(self.num_bins),
        )

    def exact_train_ic(self, num_ratios=20):
        base = len(self.bayes_hmm.observation_dist.enumerate_support())
        data = torch.stack(
            [num_to_data(num, self.num_bins, base) for num in range(base**self.num_bins)]
        ).float()
        naive_post_dist = self.hmm.posterior_log_initial_state_dist(data)
        naive_lp = self.hmm.log_prob(data)
        _ = exact_train(self.ic, data, naive_lp, naive_post_dist, self.hmm.initial_logits, num_ratios=num_ratios)
        return naive_lp, naive_post_dist

    def train_ic(self, num_train, num_ratios=20):
        """
        Trains the :py:class:`InterruptedClassifier` with `num_train` number of
        training runs, to find the optimal threshold likelihood ratios.
        Returns the training data, having already passed it to `self.ic.train`,
        so the training data is returned just in case we want to record it.

        :param int num_train: The number of runs to train the
            :py:class:`InterruptedClassifier` with.
        :returns: A :py:class:`HMMOutput` object containing

            .states: :py:class:`torch.Tensor` int
            The actual states which generated the data in `.observations`.

                shape ``(num_train, num_bins)``

            .observations: :py:class:`torch.Tensor` float.
            The data which was used to train the classifier.

                shape ``(num_train, num_bins)``

        .. seealso:: method
            :py:meth:`InterruptedClassifier.train`
        """
        hmm_out = self.hmm.sample((num_train, self.num_bins))
        states = hmm_out.states
        observations = hmm_out.observations
        ground_truth = states[:, 0]
        _ = train(self.ic, observations, ground_truth, self.bayes_hmm.initial_logits.shape[-1], num_ratios=num_ratios)
        return hmm_out

    def exact_simulation(self, return_raw=True):
        """
        Computes the data required to compute the exact misclassification rates
        of the three models under consideration, the HMM with permtuations,
        the HMM without permtuations, and the InterruptedClassifier.

        :returns: :py:class:`ExactResults` with components

            .interrupted_postprocessor: :py:class:`InterruptedExactPostprocessor`.
            Use it to compute the
            misclassification rates of the :py:class:`InterruptedClassifier`.

            .naive_postprocessor: :py:class:`PostDistExactPostprocessor`.
            Use it to compute the misclassification rates of the HMM classifier
            without permutations.

            .bayes_postprocessor: :py:class:`PostDistExactPostprocessor`.
            Use it to compute the misclassification rates of the HMM classifier
            with permutations.

            .runs: :py:class:`ExactRun`
            All the data which was used to produce the objects.
            returned just in case we would like to save it for later. See
            :py:meth:`BernoulliSimulator._exact_single_run`
            for details on members.
        """
        if self.ic.ratio is None:
            raise ValueError("Must train interrupted classifier first. Use .exact_train_ic")
        base = len(self.bayes_hmm.observation_dist.enumerate_support())
        data = torch.stack(
            [num_to_data(num, self.num_bins, base) for num in range(base**self.num_bins)]
        ).float()
        naive_lp = self.hmm.log_prob(data)
        naive_dist = self.hmm.posterior_log_initial_state_dist(data)
        bayes_result = self.bayes_hmm.get_perms(data, save_history=return_raw)
        if return_raw:
            perm = bayes_result.optimal_perm
        else:
            perm = bayes_result
        bayes_lp = self.bayes_hmm.log_prob_with_perm(data, perm)
        bayes_plisd = \
            self.bayes_hmm.posterior_log_initial_state_dist(data, perm)
        interrupted_results = self.ic.classify(data)
        if return_raw:
            results = ExactRun(
                data,
                bayes_result,
            )
        naive_ep = PostDistExactPostprocessor(
            naive_lp,
            naive_dist,
            self.bayes_hmm.initial_logits,
            self.testing_states,
        )
        bayes_ep = PostDistExactPostprocessor(
            bayes_lp,
            bayes_plisd,
            self.bayes_hmm.initial_logits,
            self.testing_states,
        )
        ip = InterruptedExactPostprocessor(
            naive_lp,
            naive_dist,
            self.bayes_hmm.initial_logits,
            self.testing_states,
            interrupted_results,
        )
        if return_raw:
            return ExactResults(
                ip,
                naive_ep,
                bayes_ep,
                results,
            )
        else:
            return ExactNoResults(
                ip,
                naive_ep,
                bayes_ep,
            )

    def empirical_simulation(self, num_samples, return_raw=True):
        """
        Computes the data required to compute the empirical misclassification
        rates of the three models under consideration, the HMM with
        permutations, the HMM without permutations, and the
        InterruptedClassifier.

        :returns: A :py:class:`EmpiricalResults` containing:

            .interrupted_postprocessor: :py:class:`InterruptedEmpiricalPostprocessor`.
            Use it to compute the
            misclassification rates of the :py:class:`InterruptedClassifier`.

            .naive_postprocessor: :py:class:`PostDistEmpiricalPostprocessor`.
            Use it to compute the misclassification rates of the HMM classifier
            without permutations.

            .bayes_postprocessor: :py:class:`PostDistEmpiricalPostprocessor`.
            Use it to compute the misclassification rates of the HMM classifier
            with permutations.

            .runs: :py:class:`EmpiricalRun`
            All the data which was used to produce the objects.
            returned just in case we would like to save it for later. See
            :py:meth:`BernoulliSimulator._single_sampled_simulation`
            for details on members.

        .. seealso:: method
            :py:meth:`BernoulliSimulator._single_sampled_simulation`
        """
        if self.ic.ratio is None:
            raise ValueError("Must train interrupted classifier first with .train_ic")
        bayes_output = \
            self.bayes_hmm.sample_min_entropy((num_samples, self.num_bins))
        naive_output = self.hmm.sample((num_samples, self.num_bins))
        naive_data = naive_output.observations
        naive_plisd = self.hmm.posterior_log_initial_state_dist(naive_data)
        interrupted_results = self.ic.classify(naive_data)
        results = EmpiricalRun(
            HMMOutPostDist(
                naive_output,
                naive_plisd,
            ),
            bayes_output,
        )
        naive_ep = PostDistEmpiricalPostprocessor(
            results.naive.hmm_output.states[..., 0],
            self.testing_states,
            self.bayes_hmm.initial_logits.shape[-1],
            results.naive.log_post_dist,
        )
        bayes_ep = PostDistEmpiricalPostprocessor(
            results.bayes.states[..., 0],
            self.testing_states,
            self.bayes_hmm.initial_logits.shape[-1],
            results.bayes.history.partial_post_log_init_dists[..., -1, :],
        )
        ip = InterruptedEmpiricalPostprocessor(
            results.naive.hmm_output.states[..., 0],
            self.testing_states,
            self.bayes_hmm.initial_logits.shape[-1],
            *interrupted_results
        )
        if return_raw:
            return EmpiricalResults(
                ip,
                naive_ep,
                bayes_ep,
                results,
            )
        else:
            return EmpiricalNoResults(
                ip,
                naive_ep,
                bayes_ep,
            )
