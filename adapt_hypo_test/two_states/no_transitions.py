r"""Main module that computes the optimal policy.

"""
import numpy as np

from adapt_hypo_test.two_states import util
from adapt_hypo_test.two_states.util import (nx_to_log_odds, m_to_r, pq_to_m, x_grid, lp_grid, log_p_log_q_to_m)
from scipy.special import logsumexp


def nop_reward(log_cond_reward, m, lp):
    r"""Computes
    :math:`\mathbb{P}(\hat{S}_{k-1} = S_{k-1}|y^{k-2},\sigma^{k-2},\sigma_{k-1}=e)`

    .. math::

        \mathbb{P}(\hat{S}_{k-1} = S_{k-1}|y^{k-2},\sigma^{k-2},\sigma_{k-1}=e) = \sum_{s_{k-1}}\mathbb{P}(\hat{S}_{k-1}=s_{k-1}|s_{k-1},y^{k-2},\sigma^{k-2},\sigma_{k-1}=e)\mathbb{P}(s_{k-1}|y^{k-2}\sigma_{k-2})

    :param log_cond_reward: :math:`\log(\mathbb{P}(\hat{S}_k=s_k|s_k,y^{k-1},\sigma^{k-1}))`
        This is an array of shape ``(2, 2k-1, 2k-1)``, where the last two
        indices correspond to the grid of posterior log odds. That is, if the
        posterior log odds

        .. math::

            x(y^{k-1}) = \log(\mathbb{P}(s_k=1|y^{k-1},\sigma^{k-1})/\mathbb{P}(s_k=0|y^{k-1},\sigma^{k-1}))

        is such that :math:`x(y^{k-1}) = n_0 r_0 + n_1 r_1`, then::

            log_cond_reward[s_k, n_0+i-1, n_1+i-1]

        is the corresponding :math:`\log(\mathbb{P}(\hat{S}_k=s_k|s_k,y^{k-1},\sigma^{k-1}))`
    :param m: :math:`\log(\mathbb{P}(y|s))`
        Should be of the form::

            m = np.log(np.array([[1-p, p],[q,1-q]]))
    :param lp: Shape
        ``(2, 2(k-2)+1, 2(k-2)+1)``. Given a log odds :math:`x = r_0 n_0 + r_1 n_1`,
        we should have ``lp[s_{k-1}, n_0+(k-2), n_1-(k-2)]`` be the corresponding
        log probability :math:`\log(\mathbb{P}(s_{k-1}|y^{k-2}, \sigma^{k-2}))`.
    :return: :math:`\mathbb{P}(\hat{S}_{k-1} = S_{k-1}|y^{k-2},\sigma^{k-2},\sigma_{k-1}=e)`,
        with shape ``(2(k-2)+1, 2(k-2)+1)``.
    """
    # Get :math:`\mathbb{P}(\hat{S}_{k-1}=s_{k-1}|s_{k-1},y^{k-2},\sigma^{k-2},\sigma_{k-1}=e)`
    nop = nop_cond_reward(log_cond_reward, m)
    nop = logsumexp(lp + nop, axis=0)
    return nop


def p_reward(log_cond_reward, m, lp):
    r"""
    Computes

    .. math::
        \mathbb{P}(\hat{S}_{k-1} = S_{k-1}|y^{k-2},\sigma^{k-2},\sigma_{k-1}=\nu) = \sum_{s_{k-1}}\mathbb{P}(\hat{S}_{k-1}=s_{k-1}|s_{k-1},y^{k-2},\sigma^{k-2},\sigma_{k-1}=\nu)\mathbb{P}(s_{k-1}|y^{k-2}\sigma_{k-2})

    See :py:func:`~adapt_hypo_test.two_states.no_transitions.nop_reward` for
    details on arguments.

    :return: :math:`\mathbb{P}(\hat{S}_{k-1} = S_{k-1}|y^{k-2},\sigma^{k-2},\sigma_{k-1}=\nu)`,
        with shape ``(2(k-2)+1, 2(k-2)+1)``. The indices correspond to the grid
        of log odds at step :math:`k-2`.
    """
    # Get :math:`\mathbb{P}(\hat{S}_{k-1}=s_{k-1}|s_{k-1},y^{k-2},\sigma^{k-2},\sigma_{k-1}=\nu)`
    p = p_cond_reward(log_cond_reward, m)
    p = logsumexp(lp + p, axis=0)
    return p


def get_actions(log_cond_reward, m):
    r"""
    Makes a grid of optimal actions, given a grid of conditional rewards
    :math:`\log(\mathbb{P}(\hat{S}_k = s_k|s_k, y^{k-1}, \sigma^{k-1}))`.

    We make this determination by performing a Bellman backup. That is, we
    compare the expected reward in the case of performing a permutation to the
    case where we do not. We return the nontrivial permutation if and only if
    the expected reward is larger if we were to.

    In equations, we return the truth value of

        .. math::

            \mathbb{P}\left( \hat{S}_0=S_0| y^{n-2},\sigma^{n-2}, \sigma_{n-1} = \nu \right)
            > \mathbb{P}\left( \hat{S}_0=S_0| y^{n-2},\sigma^{n-2}, \sigma_{n-1} = e \right)

    for all possible :math:`y^{n-2}`. As usual in this module, we instead look
    for all possible posterior log odds in place of all possible sets of data,
    to make the problem computationally tractable.

    See :py:func:`~adapt_hypo_test.two_states.no_transitions.nop_reward` for
    details on arguments.

    :return: The optimal actions at the given step, encoded as an array of shape
        ``(2*i+1, 2*i+1)``. If the returned value is ``sigma``, and the current
        posterior log odds is :math:`x = n_0 r_0 + n_1 r_1`, the optimal policy
        is to perform a permutation if and only if ``sigma[n_0+i, n_1+i]`` is
        ``True``.
    """
    r = m_to_r(m)
    k = log_cond_reward.shape[-1]
    k = k//2
    lp = lp_grid(k - 1, r)

    nop = nop_reward(log_cond_reward, m, lp)
    p = p_reward(log_cond_reward, m, lp)
    sigma = np.logical_and(p > nop, np.logical_not(np.isclose(p, nop)))
    return sigma


def nop_cond_reward(log_cond_reward, m):
    r"""
    Given :math:`\log(\mathbb{P}(\hat{S}_k = s_k|s_k, y^{k-1}, \sigma^{k-1}))`,
    computes
    :math:`\log(\mathbb{P}(\hat{S}_{k-1} = s_{k-1}|s_{k-1}, y^{k-2}, \sigma^{k-2}, \sigma_{k-1}=e))`
    according to

        .. math::

            \sum_{y_k} \mathbb{P}(\hat{S}_{k}=s_{k}|s_{k}=\sigma_{k-1}(s_{k-1}),y^{k-1},\sigma^{k-1}) \mathbb{P}(y_{k-1}|s_{k-1}, \sigma_{k-1}=e)

    Defining :math:`\chi_k^{s_k}(x_k) = \mathbb{P}(\hat{S}_k = s_k|s_k, y^{k-1}, \sigma^{k-1})`,
    we compute :math:`\chi_{k-1}^{s_{k-1}}(x_{k-1})` from
    :math:`\sum_{y_{k-1}}\mathbb{P}(y_{k-1}|\sigma_{k-1}(s_{k-1}))\chi_k^{s_k=\sigma_{k-1}(s_{k-1})}((-1)^{\sigma_{k-1}}x_{k-1}+(-1)^{y_{k-1}+1}r_{y_{k-1}})`
    where for this function, :math:`\sigma_{k-1} = e` is the identity permutation,
    and :math:`(-1)^e = 1`

    See :py:func:`~adapt_hypo_test.two_states.no_transitions.nop_reward` for
    details on arguments.

    :return: If the returned value is ``nop``, and the current posterior log
        odds is :math:`x = n_0 r_0 + n_1 r_1`, ``nop[s_k, n_0+k-1, n_1+k-1]`` is the
        value of :math:`\log(\mathbb{P}(\hat{S}_{k-1} = s_{k-1}|s_{k-1}, y^{k-2}, \sigma^{k-2}, \sigma_{k-1}=e))`.
    """
    nop = np.clip(np.logaddexp(
        # For :math:`y_{k-1} = 0`, we have
        # :math:`x_k = x_{k-1} - r_0`. Therefore shift down in the 1 direction.
        # The 2 direction doesn't change, but the possible values are from
        # :math:`n_1 = -(k-1)` to :math:`n_1 = (k-1)`, so slice into the 2 axis.
        m[:, 0][:, None, None] + log_cond_reward[:, :-2, 1:-1],
        # Similarly for :math:`y_{k-1} = 1`.
        m[:, 1][:, None, None] + log_cond_reward[:, 1:-1, 2:]
    ), None, 0.)
    return nop


def p_cond_reward(log_cond_reward, m):
    r"""
    Given :math:`\mathbb{P}(\hat{S}_k = s_k|s_k, y^{k-1}, \sigma^{k-1})`,
    computes
    :math:`\mathbb{P}(\hat{S}_{k-1} = s_{k-1}|s_{k-1}, y^{k-2}, \sigma^{k-2}, \sigma_{k-1}=\nu)`
    according to
    :math:`\sum_{y_k} \mathbb{P}(\hat{S}_{k}=s_{k}|s_{k}=\sigma_{k-1}(s_{k-1}),y^{k-1},\sigma^{k-1}) \mathbb{P}(y_{k-1}|s_{k-1}, \sigma_{k-1}=\nu)`.

    Defining :math:`\chi_k^{s_k}(x_k) = \mathbb{P}(\hat{S}_k = s_k|s_k, y^{k-1}, \sigma^{k-1})`,
    we compute :math:`\chi_{k-1}^{s_{k-1}}(x_{k-1})` from
    :math:`\sum_{y_{k-1}}\mathbb{P}(y_{k-1}|\sigma_{k-1}(s_{k-1}))\chi_k^{s_k=\sigma_{k-1}(s_{k-1})}((-1)^{\sigma_{k-1}}x_{k-1}+(-1)^{y_{k-1}+1}r_{y_{k-1}})`
    where for this function, :math:`\sigma_{k-1} = \nu` is the nontrivial permutation.
    and :math:`(-1)^\nu = -1`

    See :py:func:`~adapt_hypo_test.two_states.no_transitions.nop_reward` for
    details on arguments.

    :return: If the returned value is ``p``, and the current posterior log
        odds is :math:`x = n_0 r_0 + n_1 r_1`, ``p[s_k, n_0+k-1, n_1+k-1]`` is the
        value of :math:`\log(\mathbb{P}(\hat{S}_{k-1} = s_{k-1}|s_{k-1}, y^{k-2}, \sigma^{k-2}, \sigma_{k-1}=\nu))`.
    """
    # :math:`s_{k} = \sigma_{k-1}(s_{k-1})`, so flip the 0 axis.
    # :math:`x_{k} = (-1)x_{k-1} - (-1)^{y_{k-1}}r_{y_{k-1}}, so flip the 1 and 2 axes,
    # then apply offsets below.
    flip_reward = np.flip(log_cond_reward, axis=(0, 1, 2))
    # :math:`s_{k} = \sigma_{k-1}(s_{k-1})`, so flip the 0 axis.
    flip_m = np.flip(m, axis=0)
    p = np.clip(np.logaddexp(
        # If :math:`y_{k-1} = 0`, :math:`x_k = (-1)x_{k-1} - r_0`, so that
        # :math:`x_{k-1} + r_0 = -x_k`. Therefore shift up in the 1 axis.
        # The 2 direction doesn't change, but the possible values are from
        # :math:`n_1 = -(k-1)` to :math:`n_1 = (k-1)`, so slice into the 2 axis.
        flip_m[:, 0][:, None, None] + flip_reward[:, 2:, 1:-1],
        # Similarly for :math:`y_{k-1} = 1`.
        flip_m[:, 1][:, None, None] + flip_reward[:, 1:-1, :-2]
    ), None, 0.)
    return p


def backup_cond_reward(log_cond_reward, sigma, m):
    r"""From :math:`\log(\mathbb{P}(\hat{S}_k = s_k|s_k, y^{k-1}, \sigma^{k-1}))` and
    the optimal permutations :math:`\sigma_{k-1}`, computes
    :math:`\log(\mathbb{P}(\hat{S}_{k-1} = s_{k-1}|s_{k-1}, y^{k-2}, \sigma^{k-2}))`.

    See :py:func:`~adapt_hypo_test.two_states.no_transitions.nop_reward` for
    details on other arguments.

    :param sigma: A ``(2*k-1, 2*k-1)`` array with the optimal permutations at
        step :math:`k-1` corresponding to the grid of log odds.
    :return: An array of shape ``(2*(k-2)+1, 2*(k-2)+1)`` corresponding to
        :math:`\log(\mathbb{P}(\hat{S}_{k-1} = s_{k-1}|s_{k-1}, y^{k-2}, \sigma^{k-2}))`,
        with indices corresponding to the grid of log odds.
    """
    nop = nop_cond_reward(log_cond_reward, m)
    b_reward = nop
    p = p_cond_reward(log_cond_reward, m)
    b_reward[:, sigma] = p[:, sigma]
    return b_reward


def chi_base(n, r):
    r"""The start of the computation. This initializes an array of probability
    of correct inference after all the data has been seen, conditional on the
    true final state. We work backwards from here to obtain the optimal policy.

    :param n: Number of steps to compute for.
    :param r: The parameters
        :math:`r_0 = \log((1-p)/q)` and :math:`r_1 = \log((1-q)/p)`,
        as a numpy array of shape (2,).
    :return:
        :math:`\log(\mathbb{P}(\hat{S}_{n+1} = s_{n+1}|s_{n+1}, y^n, \sigma^n))`
        The dimensions of the returned array are
        0: :math:`s_{n+1}`, 1: :math:`n_0`, 2: :math:`n_1`
    """
    nx = x_grid(n)
    lo = nx_to_log_odds(nx, r)
    ret = np.zeros((2, 2 * n + 1, 2 * n + 1))
    ret[0][lo > 0] = -float('inf')
    ret[1][lo < 0] = -float('inf')
    ret[0][lo == 0] = np.log(.5)
    ret[1][lo == 0] = np.log(.5)
    return ret


def solve(p, q, n, log=False):
    r"""Solves for the optimal permutations.

    For the system with two states and two outputs and a trivial transition
    matrix, this solves for the optimal permutations to apply. See
    :py:mod:`adapt_hypo_test` for details.

    :param p: :math:`\mathbb{P}(y = 1|s = 0)` unless ``log`` is specified,
        in which case this should be :math:`\log(\mathbb{P}(y = 1|s = 0))`.
    :param q: :math:`\mathbb{P}(y = 0|s = 1)` unless ``log`` is specified,
        in which case this should be :math:`\log(\mathbb{P}(y = 0|s = 1))`.
    :param n: Number of steps of the hypothesis testing scenario.
    :param log: Defaults to ``False``. If ``True``, the input ``p`` and ``q``
        will be interpreted as being in log space.
    :return: A 2-tuple, containing

        The optimal permutations ``sigma``, encoded as a list of 2D arrays.
        The arrays are indexed by $n_0, n_1$, so that if the posterior log odds
        after ``i`` steps is ``r_0 n_0 + r_1 n_1``, the optimal permutation to
        apply is ``sigma[i][n_0+i, n_1+i]``. The entries of the arrays are 0 or 1,
        with 0 indicating the trivial permutation, and 1 indicating the
        nontrivial permutation.

        The conditional value function as an array of shape (2, 1, 1). This is

            .. math::

                \log(\chi_i^{s_0}) = \log(\mathbb{P}(\hat{S}_{0} = s_{0}|s_{0}))

        with the 0 index being :math:`s_0`. To obtain the probability of correct
        inference of initial state, compute

            >>> from scipy.special import logsumexp
            >>> logsumexp(chi.ravel() + log_prior)

        where chi is the second value returned from this function, and
        ``log_prior`` is the prior probability of the intial states.
    """
    if log:
        m = log_p_log_q_to_m(p, q)
    else:
        m = pq_to_m(p, q)
    r = m_to_r(m)
    chi = chi_base(n, r)
    sigmas = []
    for i in range(n):
        sigma = get_actions(chi, m)
        sigmas.append(sigma)
        chi = backup_cond_reward(chi, sigma, m)
    sigmas = list(reversed(sigmas))
    return sigmas, chi


def evaluate_sigma(sigma, x):
    p = util.index_with_nx(sigma, x)
    xx = np.copy(x)
    xx[p] = -xx[p]
    return xx, p
