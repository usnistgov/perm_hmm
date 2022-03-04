"""For the special case of two states and two outcomes, computes the optimal
permutations for the related HMM that has transition matrix equal to the
identity matrix.

Because there are only two states, we adopt the convention that the two states
are called the ``dark`` and ``bright`` states. The ``dark`` state is the one
such that the outcome is more likely to be ``0``, and the ``bright`` state is
the one such that the outcome is more likely to be ``1``.

This module uses the :py:mod:`~adapt_hypo_test` package to compute the optimal
permutations.
"""
import torch
from perm_hmm.policies.policy import PermPolicy
from adapt_hypo_test.two_states import no_transitions as nt


class IgnoreTransitions(PermPolicy):
    r"""Ignoring the transition matrix, computes the optimal permutations for
    the HMM for all possible outcomes.

    This method of computing permutations has complexity O(t**2), where t is the
    number of steps.

    In addition to the attributes of the base class, instances of this class
    have the following attributes:

    ``p``:
        A float, the probability of the dark state giving outcome 1.

    ``q``:
        A float, the probability of the bright state giving outcome 0.

    ``dtb``:
        The permutation that takes the dark state to the bright state.

    ``id``:
        The identity permutation.

    ``x``:
        A representation of the log odds of the belief state that we compute
        the permutations at. See the :py:mod:`~adapt_hypo_test` module for more
        details.

    ``sigmas``:
        A list indicating whether to apply to nontrivial permutation when
        reaching a particular log odds.
    """

    def __init__(self, possible_perms, p, q, dark_state, bright_state, save_history=False):
        r"""Initialization.

        This class computes the optimal permutations for the case that the
        transition matrix is trivial, and that there is one bright state and
        one dark state. The "true" model may have more states, and a nontrivial
        transition matrix. To make the identification between the two models,
        we need to know which state is to be interpreted as the dark state
        and which as the bright state. The possible perms of the true model are
        needed to identify which corresponds to the dark-bright swap.

        :param possible_perms: Possible permutations of the true model.
        :param dark_state: Which state of the true model corresponds to the
            dark state.
        :param bright_state: Similar for bright state.
        :raises ValueError: If the identity or the swap permutations are not
            included as possible permutations.
        """
        super().__init__(possible_perms, save_history=save_history)
        self.p = p
        self.q = q
        num_states = possible_perms.shape[-1]
        dtb = torch.nonzero(possible_perms[:, dark_state] == bright_state, as_tuple=False)
        if len(dtb) == 0:
            raise ValueError("Need to be able to take dark to bright")
        self.dtb = possible_perms[dtb[0].item()]
        identity = torch.nonzero(torch.all(possible_perms == torch.arange(num_states), dim=-1), as_tuple=False)
        if len(identity) == 0:
            raise ValueError("The identity must be an allowed permutation")
        self.id = possible_perms[identity[0].item()]
        self.x = None
        self.sigmas = None
        self.step = 0

    def reset(self, save_history=False, reset_sigmas=False):
        super().reset(save_history=save_history)
        self.x = None
        self.step = 0
        if reset_sigmas:
            self.sigmas = None

    def solve(self, n):
        r"""Needs to be called before ``calculate_perm``.

        Solves for the ideal permutations in the model where we ignore
        transitions. Calls
        :py:func:`~adapt_hypo_test.two_states.no_transitions.solve` to do so.

        :param n: The number of steps to compute for.
        :return: The expanded value function :math:`\chi`. See
            :py:mod:`~adapt_hypo_test` for more details.
        """
        self.sigmas, chi = nt.solve(self.p, self.q, n)
        return chi

    def calculate_perm(self, data: torch.Tensor) -> (torch.Tensor, dict):
        if self.sigmas is None:
            raise RuntimeError("Call .solve first with a total number of steps.")
        if self.x is None:
            self.x = torch.zeros(data.shape + (2,), dtype=int)
        self.x[~data.int().bool(), 0] -= 1
        self.x[data.int().bool(), 1] += 1
        self.step += 1
        if self.step == len(self.sigmas):
            return self.id.expand(data.shape + self.id.shape).clone().detach(), {"x": self.x.clone().detach()}
        else:
            self.x, p = nt.evaluate_sigma(self.sigmas[self.step], self.x.numpy())
            self.x = torch.from_numpy(self.x)
            perm = self.id.expand(data.shape + self.id.shape).clone().detach()
            perm[p] = self.dtb
            return perm, {"x": self.x.clone().detach()}
