import torch
from perm_hmm.return_types import PermWithHistory


class PermSelector(object):
    """
    A description of what an algorithm which selects permutations
    should do, at a minimum.

    A real algorithm should include a model, and presumably
    a calibration method.
    """

    def __init__(self, possible_perms):
        n_perms, n_states = possible_perms.shape
        if not (possible_perms.long().sort(-1).values ==
                torch.arange(n_states, dtype=torch.long).expand(
                    (n_perms, n_states)
                )).all():
            raise ValueError("The input permutations are not permutations of "
                             "the integers [0, ..., n_states]")
        self.possible_perms = possible_perms
        self.history = None

    def perm(self, data: torch.Tensor, save_history=False):
        """
        Takes a (vectorized) input of data from a single time step,
        and returns a (correspondingly shaped) permutation.
        :param torch.Tensor data: Data from the HMM.
            shape ``sample_shape + batch_shape + hmm.observation_dist.event_shape``
        :param save_history: A flag indicating whether or not to
            save the history of the computation involved to produce the
            permutations. The function shouldn't return anything
            different even if this flag is true, but the history should be
            available in the .history attribute at the end of the run.
        :return: The permutation to be applied at the next time step.
            shape ``(n_batches, n_states)``
        """
        raise NotImplementedError

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
