r"""Provides utility functions for the computation of optimal policies for two
states, two outcomes and trivial transition matrix.
"""
#%%
import itertools
import numpy as np
from scipy.special import softmax, expm1, log1p


#%%
def log_odds_to_log_probs(x):
    r"""Converts log odds to log probs.

        .. math::

            \log(p/(1-p)) \mapsto (\log(p), \log(1-p))

    This is performed by

        .. math::

            \operatorname{log\_softmax}([-x, x]/2)

    where :math:`x = \log(p/(1-p))`.

    :param x: The log odds to convert. Has arbitrary shape.
    :return: The corresponding log probs. The new axis is in the 0 position.
    """
    return np.log(softmax(np.stack([-x, x])/2, axis=0))


def nx_to_log_odds(nx, r):
    r"""Given ``nx``, return the corresponding log odds.

    The ``nx`` are pairs of integers that encode the log odds. The encoding is
    :math:`x = n_0 r_0 + n_1 r_1`

    :param nx: Pairs of integers :math:`n_0, n_1`. In particular, this is an
        array of shape ``arb. + (2,)``.
    :param r: Pairs of real numbers. Must have shape that broadcasts with
        ``nx``.
    :return: ``np.dot(nx, r)``
    """
    return np.dot(nx, r)


def nx_to_log_probs(nx, r):
    r"""Given ``nx``, return the corresponding log probs.

    The ``nx`` are pairs of integers that encode the log odds. The encoding is
    :math:`x = n_0 r_0 + n_1 r_1`

    See :py:func:`~adapt_hypo_test.two_states.util.nx_to_log_odds` for details
    on arguments.

    :return: The log probabilities, with the axis 0 corresponding to the two
        states. In particular, we return
        ``log_odds_to_log_probs(nx_to_log_odds(nx, r))``
    """
    return log_odds_to_log_probs(nx_to_log_odds(nx, r))


def r_to_logp_logq(r):
    r"""
    Given the parameters
    :math:`r_0 = \log((1-p)/q)` and :math:`r_1 = \log((1-q)/p)`,
    computes :math:`\log p` and :math:`\log q`.

    :param r: A pair of real numbers. Must destructure as
        ``r0, r1 = r``.
    :return: ``np.array([lp, lq])``, where ``lp, lq`` are
        :math:`\log(p), \log(q)` respectively.
    """
    r0, r1 = r
    norm = np.log(expm1(r0+r1))
    lp = np.log(expm1(r0)) - norm
    lq = np.log(expm1(r1)) - norm
    return np.array([lp, lq])


def m_to_r(m):
    r"""
    Given log probs, computes the parameters
    :math:`r_0 = \log((1-p)/q)` and :math:`r_1 = \log((1-q)/p)`.

    The matrix ``m`` should encode the log probabilities as::

        m = np.log(np.array([[1-p, p],[q,1-q]]))

    :return: A numpy array with two elements,
        :math:`r_0 = \log((1-p)/q)` and :math:`r_1 = \log((1-q)/p)`
        in that order.
    """
    return np.array([m[0, 0] - m[1, 0], m[1, 1] - m[0, 1]])


def pq_to_m(p, q):
    r"""
    Computes the matrix of log probabilities. Imposes the restriction that
    :math:`p+q \le 1` and :math:`p \le q`.

    :param p: A number between 0 and 1. In the hypothesis testing problem,
        This number is :math:`\mathbb{P}(y=1|s=0)`.
    :param q: Another number between 0 and 1. In the hypothesis testing problem,
        This number is :math:`\mathbb{P}(y=0|s=1)`.
    :return: ``m = np.log(np.array([[1-p, p],[q,1-q]]))``
    :raises ValueError: If one of the constraints
        :math:`p+q \le 1` or :math:`p \le q` is violated.
    """
    if p + q > 1:
        raise ValueError("We should have the condition that p + q < 1. Please"
                         " reencode the states and outcomes so that this "
                         "constraint holds. "
                         "Got {} for p and {} for q.".format(p, q))
    if p > q:
        raise ValueError("We should have p < q. Please reencode the states and"
                         "outcomes so that this constraint holds. "
                         "Got {} for p and {} for q.".format(p, q))
    m = np.log(np.array([[1-p, p],[q,1-q]]))
    return m


def log1mexp(lp):
    r"""Missing from ``scipy.special``. Computes ``log(1-exp(lp))``.

    This is useful when we want to compute :math:`1-p` for a probability
    :math:`p` encoded in log space.

    :param lp: The log probability. Arbitrary shape numpy array.
    :return: ``log(1-exp(lp))``.
    """
    lp = np.array(lp)
    mask = lp > np.log(.5)
    retval = np.empty_like(lp)
    retval[mask] = log1p(-np.exp(lp[mask]))
    retval[~mask] = np.log(-expm1(lp[~mask]))
    return retval


def log_p_log_q_to_m(lp, lq):
    r"""
    Computes the matrix of log probabilities.

    :param lp: A number. :math:`\log(p)`.
    :param lq: :math:`\log(q)`.
    :return: ``m = np.log(np.array([[1-p, p],[q,1-q]]))``
    :raises ValueError: If one of the constraints
        :math:`p+q \le 1` or :math:`p \le q` is violated.
    """
    if np.logaddexp(lp, lq) > 0:
        raise ValueError("We should have the condition that p + q < 1. Please"
                         " reencode the states and outcomes so that this "
                         "constraint holds. "
                         "Got {} for p and {} for q.".format(np.exp(lp), np.exp(lq)))
    if lp > lq:
        raise ValueError("We should have p < q. Please reencode the states and"
                         "outcomes so that this constraint holds. "
                         "Got {} for p and {} for q.".format(np.exp(lp), np.exp(lq)))
    m = np.stack([[log1mexp(lp), lp], [lq, log1mexp(lq)]])
    return m


def all_bitstrings(n):
    r"""All bitstrings of length ``n``.

    :param n: An integer
    :return: A numpy array of shape ``(2**n, n)``.
    """
    return np.array([list(i) for i in itertools.product([0, 1], repeat=n)], dtype=int)


def x_grid(k):
    r"""
    Computation starts here. We make a grid of integers corresponding to the
    log odds.

    We return a grid of :math:`(n_0, n_1)` so that we can make the corresponding
    grid of log odds as :math:`x = n_0 r_0 + n_1 r_1`, where :math:`r_0` and
    :math:`r_1` are the parameters of the model, given by :math:`\log((1-q)/p)`
    and :math:`log((1-q)/p)`, respectively.

    The point here is that Bayes' rule reduces to the following:
    :math:`x_{i+1} = x_i' \pm r_{0,1}`, where :math:`x_i'` is the log odds after
    applying the permutation. We take the plus sign and the subscript 1 if
    :math:`y_{i+1} = 1`, and the minus sign and the subscript 0 if
    :math:`y_{i+1} = 0`.

    We return nx[k+i, k+j] = [i, j]; the grid is of size (2*k+1, 2*k+1, 2).
    Use nx_to_log_odds to convert to log odds.

    :param k: An integer.
    :return: An array of shape ``(2*k+1, 2*k+1, 2)``.
    """
    nx = np.moveaxis(np.stack(np.meshgrid(np.arange(-k, k + 1), np.arange(-k, k + 1))), (0, 1, 2), (2, 1, 0))
    return nx


def lp_grid(k, r):
    r"""Grid of log probabilities.

    First we construct a ``(2*k+1, 2*k+1)`` grid of log odds
    :math:`\log\left(\frac{\mathbb{P}(s=1|y^k)}{\mathbb{P}(s=0|y^k)}\right)`,
    those of the form
    :math:`n_0 r_0 + n_1 r_1`, with :math:`n_0, n_1 \in \{-k, \ldots, k\}`.
    Then we convert these log odds to posterior log probabilities
    :math:`\log(\mathbb{P}(s|y^k))`, so that the
    returned array has shape ``(2, 2*k+1, 2*k+1)``. The 0 axis is the state,
    while the others are the corresponding indices of the grid of log odds.

    :param int k: Sets the size of the grid.
    :param r: numpy array, shape ``(2,)``. These are :math:`r_0 = \log((1-p)/q)`
        and :math:`r_1 = \log((1-q)/p)` in that order.
    :return: The grid of posterior log probabilities.
    """
    nx = x_grid(k)
    lp = nx_to_log_probs(nx, r)
    return lp


def index_with_nx(a, nx):
    if not a.shape[0] == a.shape[1]:
        raise ValueError("Array to index into should be square in the first two dimensions")
    if not a.shape[0] % 2 == 1:
        raise ValueError("Array to index into should be of odd length in the first two dimensions")
    k = a.shape[0]//2
    inds = tuple(np.moveaxis(nx, (-1,), (0,)) + k)
    return a[inds]

