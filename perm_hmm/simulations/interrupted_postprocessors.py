import torch
import pyro

import perm_hmm.return_types
from perm_hmm.simulations.postprocessing import ExactPostprocessor, EmpiricalPostprocessor


class InterruptedExactPostprocessor(ExactPostprocessor):
    """
    Processes data from a numerical experiment which computes the exact
    misclassification rate for the interrupted classifier.
    """

    def __init__(self, log_prob, log_post_dist, log_prior_dist, testing_states,
                 classifications, break_flag=None, log_like_ratio=None):
        """
        Initializes the postprocessor.

        :param bayes_perm_hmm.return_types.ClassBreakRatio class_break_ratio:
            Contains:

            :param torch.Tensor classified_bright: bool.
                All classifications
                from the interrupted classifier at all combinations of threshold
                likelihood ratios.

                shape ``(n_runs, num_ratios, num_ratios)``

            :param torch.Tensor break_flag: bool.
                All indications of whether or
                not the classifier terminated the run early.

                shape ``(n_runs, num_ratios, num_ratios)``

            :param torch.Tensor log_like_ratio: float
                All the log likelihood ratios.

                shape ``(n_runs,)``

        See superclass for details on other inputs.
        """
        self.ratios = log_like_ratio
        if self.ratios is not None:
            score = self.ratios.abs()
        else:
            score = self.ratios
        super().__init__(
            log_prob,
            log_post_dist,
            log_prior_dist,
            testing_states,
            classifications,
            score,
        )
        self.break_flag = break_flag

    def postselect(self, postselect_mask):
        """
        Postselects the data according to a postselection mask.

        See superclass for details on inputs.

        :returns: InterruptedExactPostprocessor.
            Contains a postselected version of the data.
        """
        if (~postselect_mask).all():
            raise ValueError("Can't throw out all the data.")
        remaining_log_prob = self.log_prob[postselect_mask].logsumexp(-1)
        postselected_post_dist = self.log_post_dist[postselect_mask]
        postselected_log_prob = \
            self.log_prob[postselect_mask] - remaining_log_prob
        postselected_all_classifications = \
            self.classifications[..., postselect_mask]
        if self.break_flag is not None:
            postselected_all_break_flag = self.break_flag[..., postselect_mask]
        else:
            postselected_all_break_flag = None
        if self.break_flag is not None:
            postselected_all_ratios = self.ratios[..., postselect_mask]
        else:
            postselected_all_ratios = None
        postselected_postprocessor = InterruptedExactPostprocessor(
            postselected_log_prob,
            postselected_post_dist,
            self.log_prior_dist,
            self.testing_states,
            postselected_all_classifications,
            postselected_all_break_flag,
            postselected_all_ratios,
        )
        return postselected_postprocessor


class InterruptedEmpiricalPostprocessor(EmpiricalPostprocessor):
    """
    Processes data from a numerical experiment which computes the empirical
    misclassification rate for the interrupted classifier.
    """

    def __init__(self, ground_truth, testing_states, total_num_states, classifications, break_flag=None, log_like_ratio=None):
        """
        Initializes the postprocessor.

        Due to the varied use cases, we break out each of the elemnts of the
        :py:class:`ClassBreakRatio`. We can also initialize this as::

            InterruptedEmpiricalPostprocessor(
                actually_bright,
                actually_dark,
                *class_break_ratio
            )

        where ``class_break_ratio`` is a :py:class:`ClassBreakRatio`

        :param torch.Tensor actually_bright: bool.
            Ground truth of which runs were actually bright.

            shape ``(n_runs,)``

        :param torch.Tensor actually_dark: bool.
            Ground truth of which runs were actually dark.

            shape ``(n_runs,)``

        :param torch.Tensor classified_bright: bool.
            Indicates whether the run was classified bright.

            shape ``(n_runs,)``

        :param torch.Tensor break_flag: bool.
            Indicates whether the run was
            terminated before the full detection cycle.

            shape ``(n_runs,)``

        :param torch.Tensor log_like_ratio: float.
            The final log likelihood ratio of the runs.

            shape ``(n_runs,)``
        """
        if log_like_ratio is not None:
            score = log_like_ratio.abs()
        else:
            score = None
        super().__init__(ground_truth, testing_states, total_num_states, classifications, score)
        self.break_flag = break_flag
        """Indicates whether the run was interrupted."""
        self.log_like_ratio = log_like_ratio
        """The final log likelihood ratio of the runs."""

    def postselect(self, postselection_mask):
        """
        Postselects the postprocessor.

        See superclass for details.
        """
        if (~postselection_mask).all():
            raise ValueError("Can't throw out all the data.")
        return InterruptedEmpiricalPostprocessor(
            self.ground_truth[..., postselection_mask],
            self.testing_states,
            self.total_num_states,
            self.classifications[..., postselection_mask],
            self.break_flag[..., postselection_mask],
            self.log_like_ratio[..., postselection_mask],
        )


