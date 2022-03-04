"""This is an example of a very simple PermPolicy.

The :py:class:`~perm_hmm.policies.policy.PermPolicy` is a class that is
used to select a permutation based on data seen thus far. It takes care of some
boilerplate, but should be subclassed to implement the actual selection
algorithm, and done so in a particular way. This example is meant to demonstrate
how to subclass the :py:class:`~perm_hmm.policies.policy.PermPolicy` class
appropriately.

The :py:class:`~perm_hmm.policies.rotator_policy.RotatorPolicy` is a
:py:class:`~perm_hmm.policies.policy.PermPolicy` that rotates the states,
independently of the data seen thus far.
"""
import torch
from perm_hmm.policies.policy import PermPolicy


def cycles(num_states):
    """Generates a list of cycles of length num_states.
    :param int num_states: The number of states in the cycle.
    :return: A tensor containing the cycles.
    """
    return (
                   torch.arange(num_states).repeat((num_states, 1)) +
                   torch.arange(num_states).unsqueeze(-1)
           ) % num_states


class RotatorPolicy(PermPolicy):
    """A minimal example of a PermPolicy, for demonstration of subclassing the
    PermPolicy class.

    Always implements the cycle that shifts states by 1, regardless of the data.

    Has attributes:

    ``num_states``:
        The number of states in the HMM.

    ``index``:
        The number of permutations applied, modulo num_states. This is not
        necessary, but is here for demonstration purposes for the reset method.
    """

    def __init__(self, hmm, save_history=False):
        r"""Initializes the RotatorPolicy.

        Given an HMM, computes the possible permutations as the cycles on the
        states of the HMM, and initializes the policy with these cycles.

        .. seealso:: :py:meth:`~perm_hmm.policies.policy.PermPolicy.__init__`
        :param perm_hmm.models.hmms.PermutedDiscreteHMM hmm: An HMM.
        :param save_history: Whether to save the computation history. If
            specified, the history of the permutations is saved in the attribute
            ``.calc_history``.
        """
        self.index = 0
        self.num_states = hmm.initial_logits.shape[0]
        possible_perms = cycles(self.num_states)
        super().__init__(possible_perms, save_history=save_history)

    def reset(self, save_history=False):
        """This method is for resetting the PermPolicy.

        Because in general the algorithm for the selection of permutations
        can be stateful, this method is necessary to reinitialize the object
        before reusing it.

        Calling super().reset resets the history of permutations and history of
        calculation. In general one may want to reinitialize other parts of the
        PermPolicy.

        :param save_history: Indicates whether to save the computations
            involved in selecting the permutations.
        :return: None
        """
        super().reset(save_history=save_history)
        self.index = 0

    def calculate_perm(self, data: torch.Tensor) -> (torch.Tensor, dict):
        """The main method for selecting permutations.

        Given data, returns the appropriate permutation, and any useful
        information about the calculation involved in the form of a dictionary.

        :param data: The data at a single step to compute the permutation for.
        :return: A choice of permutation, and the calculation involved in
            computing that permutation.
        """
        self.index = (self.index + 1) % self.num_states
        return self.possible_perms[1].repeat((data.shape[0], 1)), {}
