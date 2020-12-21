import torch

from perm_hmm.classifiers.generic_classifiers import MAPClassifier
from perm_hmm.return_types import PermWithHistory


class PermClassifier(MAPClassifier):

    def __init__(self, model, testing_states, perm_selector):
        self.model = model
        self.testing_states = testing_states
        self.perm_selector = perm_selector

    def get_perms(self, data, obs_event_dim, save_history=False):
        r"""
        Given a run of data, returns the posterior initial state distributions,
        the optimal permutations according to
        the bayesian heuristic, and the posterior entropies.

        This should be used to precompute the permutations for a given model
        and given data sequence.

        :param torch.Tensor data: float.
            The sequence of data to compute the optimal permutations for

                shape ``batch_shape + (time_dim,)``

        :returns: A :py:class:`PermWithHistory` object with leaves

            .optimal_perm: A :py:class:`torch.Tensor` type :py:class:`int`
            containing the optimal permutations to have applied.

                shape ``batch_shape + (time_dim,)``

            .history.partial_post_log_init_dists: A :py:class:`torch.Tensor`
            containing :math:`p(s_0|y^{i})` for all :math:`i`.

                shape ``batch_shape + (time_dim, state_dim)``

            .history.expected_entropy: A :py:class:`torch.Tensor` containing
            :math:`\operatorname{min}_{\sigma}H_\sigma(S_0|Y^i, y^{i-1})`
            for all :math:`i`.

                shape ``batch_shape + (time_dim,)``

        .. seealso:: method :py:meth:`PermutedDiscreteHMM.sample_min_entropy`
        """
        d_shape = data.shape
        shape = d_shape[:len(d_shape) - obs_event_dim]
        max_t = shape[-1]
        perms = []
        for i in range(max_t):
            perms.append(self.perm_selector.perm(
                data[(..., i) + (
                    slice(None),) * obs_event_dim],
                save_history=save_history,
            ))
        perms = torch.stack(perms, -2)
        if save_history:
            return PermWithHistory(
                perms,
                self.perm_selector.history,
            )
        else:
            return perms

    def classify(self, data, obs_event_dim, verbosity=0):
        if (verbosity <= 0):
            verbosity = 0
        if verbosity >= 3:
            verbosity = 3
            save_history = True
        else:
            save_history = False
        if save_history:
            perms, history = self.get_perms(data, obs_event_dim, save_history=save_history)
        else:
            perms = self.get_perms(data, obs_event_dim, save_history=save_history)
        classifications = MAPClassifier(self.model.expand_with_perm(perms)).classify(data, obs_event_dim, verbosity)
        if verbosity == 2:
            classifications["perms"] = perms
        if verbosity == 3:
            classifications["history"] = self.perm_selector.history
        return classifications
