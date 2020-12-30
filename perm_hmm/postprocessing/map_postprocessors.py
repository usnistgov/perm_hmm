# import torch
#
# from perm_hmm.postprocessing.postprocessing import ExactPostprocessor, EmpiricalPostprocessor
# from perm_hmm.util import entropy
#
# class PostDistExactPostprocessor(ExactPostprocessor):
#     """
#     Computes statistics from the data obtained from a simulation used to compute
#     the exact misclassification rates.
#     """
#
#     def __init__(self, log_joint, testing_states, classifications, score=None):
#         """
#         See superclass for details.
#         Classifies the runs using the MAP classifier.
#         """
#         log_post_dist = log_joint - log_joint.logsumexp(-2)
#         if score is None:
#             score = -entropy(log_post_dist)
#         super().__init__(log_joint, testing_states, classifications, score)
#
#
# class PostDistEmpiricalPostprocessor(EmpiricalPostprocessor):
#     """
#     Processes data from a numerical experiment which computes the empirical
#     misclassification rate for the HMM classifier.
#     """
#
#     def __init__(self, ground_truth, testing_states, total_num_states, log_post_dist):
#         """
#         Initializes the postprocessor. Uses the maximum a posteriori estimator
#         for the initial state inference.
#
#         We use the negative posterior initial state entropy as the score.
#
#         :param torch.Tensor log_post_dist:
#             The posterior log initial state distribution of the
#             runs according the the model used to classify.
#
#             shape ``(n_runs, state_dim)``
#
#         See superclass for more details on other parameters.
#         """
#         self.log_post_dist = log_post_dist
#         """
#         The posterior log initial state distribution. A :py:class:`torch.Tensor`
#         of
#
#             shape ``(n_runs, state_dim)``
#         """
#         classifications = testing_states[self.log_post_dist[..., testing_states].argmax(-1)]
#         score = -entropy(self.log_post_dist)
#         super().__init__(ground_truth, testing_states, total_num_states, classifications, score)
#
#     def postselect(self, postselection_mask):
#         """
#         Postselects the postprocessor.
#
#         See superclass for details.
#         """
#         if (~postselection_mask).all():
#             raise ValueError("Can't throw out all the data.")
#         return PostDistEmpiricalPostprocessor(
#             self.ground_truth[..., postselection_mask],
#             self.testing_states,
#             self.total_num_states,
#             self.log_post_dist[..., postselection_mask, :],
#         )
