import torch


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
