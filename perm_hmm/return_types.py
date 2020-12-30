from typing import NamedTuple

import torch


hmm_fields = [
    ('states', torch.Tensor),
    ('observations', torch.Tensor),
]

HMMOutput = NamedTuple('HMMOutput', hmm_fields)
perm_hmm_fields = hmm_fields + [('perm', torch.Tensor)]
PermHMMOutput = NamedTuple(
    'PermHMMOutput',
    perm_hmm_fields,
)
PHMMOutHistory = NamedTuple(
    'MinEntHMMOutput',
    perm_hmm_fields + [('history', dict)]
)


class PostYPostS0(NamedTuple):
    r"""
    Contains the posterior output distribution, and the
    posterior initial distribution.

    .. seealso:: return type of :py:meth:`PermutedDiscreteHMM.full_posterior`
    """
    log_post_y: torch.Tensor
    r""":py:class:`torch.Tensor`, float.
    The posterior output distribution :math:`p(y_n | y^{n-1})`

        shape ``(n_outcomes, n_perms)``
    """
    log_post_init: torch.Tensor
    r""":py:class:`torch.Tensor`, float.
    The posterior initial state distribution :math:`p(s_0 | y^{n-1})`

        shape ``(n_outcomes, n_perms, state_dim)``
    """


class GenDistEntropy(NamedTuple):
    """
    Contains the expected posterior entropies and the log posterior
    distributions which generate them.

    .. seealso:: the return type of
        :py:meth:`PermutedDiscreteHMM.expected_entropy`
    """
    log_dists: PostYPostS0
    """:py:class:`PostYPostS0`
    The log distributions used to compute the
    posterior entropy.
    """
    expected_entropy: torch.Tensor
    """:py:class:`torch.Tensor`, float.
    The expected posterior entropy.

        shape ``(n_perms,)``
    """

