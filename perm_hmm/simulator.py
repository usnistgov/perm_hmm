"""
Simulates the initial state discrimination experiment using different
methods, to compare the resulting error rates.
"""
from typing import NamedTuple

import torch

from perm_hmm.return_types import ExactRun, \
    HMMOutPostDist, EmpiricalRun
from perm_hmm.util import num_to_data
from perm_hmm.interrupted_training import train, exact_train
from perm_hmm.postprocessing import InterruptedExactPostprocessor, \
    PostDistExactPostprocessor, InterruptedEmpiricalPostprocessor, \
    PostDistEmpiricalPostprocessor, ExactPostprocessor, EmpiricalPostprocessor

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


class HMMSimulator(object):

    def __init__(self, phmm, testing_states):
        """
        Initializes the experiment.

        :param bayes_perm_hmm.hmms.PermutedDiscreteHMM phmm:
            the model whose
            misclassification rate will be computed. The naive_hmm parameters
            will be classified from those of bayes_hmm.
        :param torch.Tensor testing_states: states to perform hypothesis tests for.
        :param int num_bins: time dimension of data to be collected.

        :raises: ValueError
        """
        self.phmm = phmm
        """:py:class:`PermutedDiscreteHMM`
        The model whose misclassification rates we wish to analyze.
        """
        num_states = len(self.phmm.initial_logits)
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

    def all_classifications(self, num_bins, classifier=None, perm_selector=None, verbosity=0, save_history=False):
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
        base = len(self.phmm.observation_dist.enumerate_support())
        data = torch.stack(
            [num_to_data(num, num_bins, base) for num in range(base**num_bins)]
        ).float()
        if perm_selector is not None:
            perm_selector.reset(save_history=save_history)
            perms = perm_selector.get_perms(data, save_history=save_history)
            if save_history:
                history = perm_selector.history
            classi_dict = classifier.classify(data, perms=perms, verbosity=verbosity)
        else:
            perms = None
            classi_dict = classifier.classify(data, verbosity=verbosity)
        if verbosity:
            classifications = classi_dict["classifications"]
        else:
            classifications = classi_dict
        lp = self.phmm.log_prob_with_perm(data, perms)
        dist = self.phmm.posterior_log_initial_state_dist(data, perms)
        ep = ExactPostprocessor(
            lp,
            dist,
            self.phmm.initial_logits,
            self.testing_states,
            classifications,
        )
        if verbosity:
            if save_history:
                return ep, classi_dict, history
            return ep, classi_dict
        return ep

    def simulate(self, num_bins, num_samples, classifier=None, perm_selector=None, verbosity=0, save_history=False):
        if perm_selector is not None:
            perm_selector.reset(save_history=save_history)
        output = self.phmm.sample((num_samples, num_bins), perm_selector=perm_selector)
        if perm_selector is not None:
            perms = perm_selector.perm_history
        else:
            perms = None
        if save_history and (perm_selector is not None):
            history = perm_selector.history
        data = output.observations
        if perms:
            classi_dict = classifier(data, perms, verbosity=verbosity)
        else:
            classi_dict = classifier(data, verbosity=verbosity)
        if verbosity:
            classifications = classi_dict["classifications"]
        else:
            classifications = classi_dict
        ep = EmpiricalPostprocessor(
            output.states,
            self.testing_states,
            len(self.phmm.initial_logits),
            classifications,
        )
        if verbosity:
            if save_history and (perm_selector is not None):
                return ep, classi_dict, history
            return ep, classi_dict
        return ep
