"""
Simulates the initial state discrimination experiment using different
methods, to compare the resulting error rates.
"""
from typing import NamedTuple

import torch

from perm_hmm.util import num_to_data
from perm_hmm.simulations.postprocessing import ExactPostprocessor, EmpiricalPostprocessor
from perm_hmm.classifiers.perm_classifier import PermClassifier

class HMMSimulator(object):

    def __init__(self, phmm):
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

    def check_states(self, testing_states):
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
        return testing_states

    def all_classifications(self, num_bins, testing_states, classifier=None, perm_selector=None, verbosity=0):
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
        self.check_states(testing_states)
        base = len(self.phmm.observation_dist.enumerate_support())
        data = torch.stack(
            [num_to_data(num, num_bins, base) for num in range(base**num_bins)]
        ).float()
        if verbosity > 1:
            save_history = True
        else:
            save_history = False
        if classifier is None:
            classifier = PermClassifier(self.phmm)
        if perm_selector is not None:
            perm_selector.reset(save_history=save_history)
            perms = perm_selector.get_perms(data, -1)
            if save_history:
                history = perm_selector.history
            classi_dict = classifier.classify(data, testing_states, perms=perms, verbosity=verbosity)
        else:
            perms = None
            classi_dict = classifier.classify(data, testing_states, verbosity=verbosity)
        if verbosity:
            classifications = classi_dict[b"classifications"]
            if save_history:
                classi_dict[b"history"] = history
        else:
            classifications = classi_dict
        lp = self.phmm.log_prob_with_perm(data, perms)
        dist = self.phmm.posterior_log_initial_state_dist(data, perms)
        ep = ExactPostprocessor(
            lp,
            dist,
            self.phmm.initial_logits,
            testing_states,
            classifications,
        )
        if verbosity:
            return ep, classi_dict
        return ep

    def simulate(self, num_bins, num_samples, testing_states, classifier=None, perm_selector=None, verbosity=0):
        self.check_states(testing_states)
        if verbosity > 1:
            save_history = True
        else:
            save_history = False
        if perm_selector is not None:
            perm_selector.reset(save_history=save_history)
        output = self.phmm.sample((num_samples, num_bins), perm_selector=perm_selector)
        if perm_selector is not None:
            perms = perm_selector.perm_history
        else:
            perms = None
        if save_history and (perm_selector is not None):
            history = perm_selector.calc_history
        data = output.observations
        if classifier is None:
            classifier = PermClassifier(self.phmm)
        if perms is not None:
            classi_dict = classifier.classify(data, testing_states, perms, verbosity=verbosity)
        else:
            classi_dict = classifier.classify(data, testing_states, verbosity=verbosity)
        if verbosity:
            classifications = classi_dict[b"classifications"]
            classi_dict[b"data"] = data
            if save_history:
                classi_dict[b"history"] = history
        else:
            classifications = classi_dict
        ep = EmpiricalPostprocessor(
            output.states[..., 0],
            testing_states,
            len(self.phmm.initial_logits),
            classifications,
        )
        if verbosity:
            return ep, classi_dict
        return ep
