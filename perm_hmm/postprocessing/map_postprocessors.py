import torch

from perm_hmm.postprocessing.postprocessing import ExactPostprocessor, EmpiricalPostprocessor
from perm_hmm.util import entropy

class PostDistExactPostprocessor(ExactPostprocessor):
    """
    Computes statistics from the data obtained from a simulation used to compute
    the exact misclassification rates.
    """

    def __init__(self, log_prob, log_post_dist, log_prior_dist, testing_states,
                 classifications, score=None):
        """
        See superclass for details.
        Classifies the runs using the MAP classifier.
        """
        if classifications is None:
            classifications = testing_states[log_post_dist[..., testing_states].argmax(-1)]
        if score is None:
            score = -entropy(log_post_dist)
        super().__init__(log_prob, log_post_dist, log_prior_dist,
                         testing_states, classifications, score)

    def postselect(self, postselect_mask):
        """
        Postselects the data according to a postselection mask.

        See superclass for details on inputs.

        :returns: PostDistExactPostprocessor. Contains a postselected version of the
            data.
        """
        if (~postselect_mask).all():
            raise ValueError("Can't throw out all the data.")
        remaining_log_prob = self.log_prob[postselect_mask].logsumexp(-1)
        postselected_post_dist = self.log_post_dist[postselect_mask]
        postselected_log_prob = \
            self.log_prob[postselect_mask] - remaining_log_prob
        postselected_postprocessor = PostDistExactPostprocessor(
            postselected_log_prob,
            postselected_post_dist,
            self.log_prior_dist,
            self.testing_states,
            self.classifications[postselect_mask],
            self.score[postselect_mask],
        )
        return postselected_postprocessor


class PostDistEmpiricalPostprocessor(EmpiricalPostprocessor):
    """
    Processes data from a numerical experiment which computes the empirical
    misclassification rate for the HMM classifier.
    """

    def __init__(self, ground_truth, testing_states, total_num_states, log_post_dist):
        """
        Initializes the postprocessor. Uses the maximum a posteriori estimator
        for the initial state inference.

        We use the negative posterior initial state entropy as the score.

        :param torch.Tensor log_post_dist:
            The posterior log initial state distribution of the
            runs according the the model used to classify.

            shape ``(n_runs, state_dim)``

        See superclass for more details on other parameters.
        """
        self.log_post_dist = log_post_dist
        """
        The posterior log initial state distribution. A :py:class:`torch.Tensor`
        of

            shape ``(n_runs, state_dim)``
        """
        classifications = testing_states[self.log_post_dist[..., testing_states].argmax(-1)]
        score = -entropy(self.log_post_dist)
        super().__init__(ground_truth, testing_states, total_num_states, classifications, score)

    def postselect(self, postselection_mask):
        """
        Postselects the postprocessor.

        See superclass for details.
        """
        if (~postselection_mask).all():
            raise ValueError("Can't throw out all the data.")
        return PostDistEmpiricalPostprocessor(
            self.ground_truth[..., postselection_mask],
            self.testing_states,
            self.total_num_states,
            self.log_post_dist[..., postselection_mask, :],
        )
