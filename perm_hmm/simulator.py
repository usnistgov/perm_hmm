"""
Simulates the initial state discrimination experiment using different
methods, to compare the resulting error rates.
"""

import torch

from perm_hmm.util import num_to_data
from perm_hmm.postprocessing import ExactPostprocessor, EmpiricalPostprocessor
from perm_hmm.classifiers.perm_classifier import PermClassifier


# TODO: Add postselection, perhaps by adding a .score method to the classifiers, and a score flag to the simulation
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

    def all_classifications(self, num_bins, classifier=None, perm_selector=None, verbosity=0):
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
            classi_dict = classifier.classify(data, perms=perms, verbosity=verbosity)
        else:
            perms = None
            classi_dict = classifier.classify(data, verbosity=verbosity)
        if verbosity:
            classifications = classi_dict[b"classifications"]
            if save_history:
                classi_dict[b"history"] = history
        else:
            classifications = classi_dict
        lp = self.phmm.log_prob_with_perm(data, perms)
        dist = self.phmm.posterior_log_initial_state_dist(data, perms)
        log_joint = dist.T + lp
        ep = ExactPostprocessor(
            log_joint,
            classifications,
        )
        if verbosity:
            return ep, classi_dict
        return ep

    def simulate(self, num_bins, num_samples, classifier=None, perm_selector=None, verbosity=0):
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
        history = None
        if save_history:
            if perm_selector is not None:
                history = perm_selector.calc_history
        data = output.observations
        if classifier is None:
            classifier = PermClassifier(self.phmm)
        if perms is not None:
            classi_dict = classifier.classify(data, perms=perms, verbosity=verbosity)
        else:
            classi_dict = classifier.classify(data, verbosity=verbosity)
        if verbosity:
            classifications = classi_dict[b"classifications"]
            classi_dict[b"data"] = data
            if history is not None:
                classi_dict[b"history"] = history
        else:
            classifications = classi_dict
        ep = EmpiricalPostprocessor(
            output.states[..., 0],
            classifications,
        )
        if verbosity:
            return ep, classi_dict
        return ep
