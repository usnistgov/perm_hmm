r"""
Computes the output probabilities of the Beryllium ion.

The transition matrix was calculated separately [Zarantonello]_

All equations from [Langer]_, Chapter 2.

This module computes the populations of the various energy levels, when
addressed by a laser resonant with the :math:`^2S_{1/2}, F=2, m_F=2
\leftrightarrow ^2P_{3/2}, m_J=3/2` level, with perfect :math:`\sigma^+`
polarization.

All physical quantities with units of time or frequency are nondimensionalized
using the laser linewidth,
:math:`\gamma = 2\pi \times 19.4 \times 10^6`

:math:`\hbar` is also set to 1.

.. [Langer] C. E. Langer, High fidelity quantum information processing with
     trapped ions, Ph.D. thesis (2006), Last updated - 2016-05-25.

.. [Zarantonello] G. Zarantonello, Private Communication (2022).
"""
import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import argparse
import scipy.special as sc
from scipy.special import logsumexp, logit, log1p, expm1
import networkx as nx
from perm_hmm.util import ZERO
import mpmath


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



def reg_upper_incomp_gamma(n, x):
    r"""The regularized upper incomplete gamma function, evaluated at positive integers.

    This function uses mpmath to evaluate the function, using an expansion of
    the expression in terms of recursion.

    :param n: An integer
    :param x: A positive real.
    :return: Upper regularized incomplete gamma at n and x.
    """
    return mpmath.polyval(mpmath.taylor(mpmath.exp, 0, (n-1))[::-1], x)*mpmath.exp(-x)



l_to_fmf = dict([
    (0, (2, -2)),
    (1, (2, -1)),
    (2, (2, 0)),
    (3, (2, 1)),
    (4, (2, 2)),
    (5, (1, 1)),
    (6, (1, 0)),
    (7, (1, -1))
])
fmf_to_l = dict([(v, k) for k, v in l_to_fmf.items()])
N_STATES = 8
BRIGHT_STATE = fmf_to_l[(2, 2)]
DARK_STATE = fmf_to_l[(1, -1)]

dimensionful_gamma = 2 * np.pi * 19.4 * 10 ** 6
dimensionful_delta_z = 233 * 10 ** 6
dimensionful_detection_laser = 2 * np.pi * 957200 * 10 ** 9
dimensionful_A = -625.0088370 * 10 ** 6
dimensionful_gamma_c = 30 / (330 * 10 ** (-6))


gi_prime = 2.13477985 * 10 ** (-4)
s0 = 1 / 2
delta_z = dimensionful_delta_z / dimensionful_gamma
A = dimensionful_A / dimensionful_gamma
gamma = dimensionful_gamma / dimensionful_gamma
gamma_prime = gamma / 2 * np.sqrt(1 + s0)
g_squared = gamma ** 2 * s0 / 8

dimensionful_bohr_magneton = 9.27401008 * 10 ** (-24)
dimensionful_magnetic_field = .01196
dimensionful_planck = 6.626 * 10 ** (-34)
gj = -2.00226206

x0 = -dimensionful_bohr_magneton * dimensionful_magnetic_field * gj * (
        1 - gi_prime) / (dimensionful_planck * dimensionful_A)
f_plus = 2
gamma_c = dimensionful_gamma_c / dimensionful_gamma
gamma_bg = .06 * gamma_c
r_bg = gamma_bg/gamma_c


def adjugate(m):
    r"""The `classical adjugate`_ of a matrix.

    Used to solve coupled first order ordinary differential equations when using
    partial fractions and the `Laplace transform`_.

    See Equation 2.25 in [Langer]_

    .. _`classical adjugate`: https://en.wikipedia.org/wiki/Adjugate_matrix
    .. _`Laplace transform`: https://en.wikipedia.org/wiki/Laplace_transform

    :param m: The matrix to take the classical adjugate of.
    :return: Its adjugate.
    :raises ValueError: If passed a non-square matrix.
    """
    n = m.shape[-1]
    if m.shape[-2:] != (n, n):
        raise ValueError("Can't take the adjugate of a non-square matrix")
    minor_slicer = np.tile(np.arange(n), (n, 1))
    minor_slicer = np.tril(minor_slicer, -1)[:, :-1] + np.triu(minor_slicer, 1)[:, 1:]
    slicer = (np.moveaxis(minor_slicer[:, :, None, None], [1, 2], [2, 1]), np.moveaxis(minor_slicer[None, None, :, :], [1, 2], [2, 1]))
    signs = -((((np.arange(n)[:, None] + np.arange(n)[None, :]) % 2) * 2) - 1)
    return np.moveaxis(np.linalg.det(m[(..., ) + slicer]), [-2, -1], [-1, -2])*signs


def m_matrix():
    r"""The matrix that describes all the transfers of population.

    This matrix was derived separately by [Zarantonello]_.

    See Equation 2.22 in [Langer]_

    :return: An 8x8 matrix describing the rates of transition. Solve

        .. math::

            \frac{\partial v}{\partial t} = M v

        to compute the population vector :math:`v` at some time.
    """
    grates = np.array([
        [-53515.203611955076,0,0,0,0,0,0,0],
        [9370.314072957095,-299751.14204706636,0,0,0,0,0,151.37759092810367],
        [0.8487196386391054,110188.69242417434,-440602.25430061953,0,0,0,291.28995219314106,20.351987799548343],
        [0,4.013292206686567,191257.83235340557,-121864.12026454191,0,347.39534056041276,85.31310792362446,9.72689850253587e-6],
        [0,0,2.436625412363273,15768.695472884541,0,264.1491164266571,0.00003137702749778637,0],
        [0,2.326867856226699,110807.8166551244,106095.42479165737,0,-611.5444569870699,50.116889724201215,5.7102154286647985e-6],
        [1.4105332956920842,182205.61618431052,138534.16866667714,0,0,0,-426.71998121799425,34.474966451766335],
        [44142.63028606365,7350.493278518571,0,0,0,0,0,-206.20456061653226],
    ])
    grates = grates / dimensionful_gamma
    return grates


def polymat(m, s):
    r"""The matrix of characteristic polynomials of minors of the resolvent of
    ``m``.

    This is the quantity :math:`p_{ij}` in Equation 2.25 in [Langer]_

    :param m: Should be passed the matrix returned by ``m_matrix``.
    :param s: The :math:`s` in Laplace space to evaluate the characteristic
        polynomial of minors at.
    :return: The matrix of characteristic polynomials evaluated at :math:`s`.
        Has shape ``(8, 8) + s.shape``.
    """
    m_shape = m.shape
    arr_s = np.array(s)
    s_shape = arr_s.shape
    reshaped_s = arr_s.reshape((-1, 1, 1))
    n = m.shape[-1]
    eye = np.eye(n)
    val = adjugate(m[..., None, :, :] - eye*reshaped_s).reshape(m_shape[:-2] + s_shape + (n, n))
    return val


def akij(m, ev=None):
    r"""The coefficients of the partial fraction decomposition of the resolvent.

    This is :math:`a_{ij}^{(k)}` in Equation 2.25 in [Langer]_

    :param m: Should be passed the matrix returned by ``m_matrix``.
    :param ev: The eigenvalues of :math:`m`. If None, will be computed.
    :return: :math:`a_{ij}^{(k)}`, with dimensions 0: :math:`k`, 1: :math:`i`,
        2: :math:`j`.
    """
    n = m.shape[-1]
    if ev is None:
        ev = np.linalg.eigvals(m)
    pij = adjugate(m[..., None, :, :] - np.eye(m.shape[-1])*ev[..., None, None])
    evprod = np.expand_dims(np.prod(ev[..., None, :] - ev[..., :, None] + np.eye(n), axis=-1), [-1, -2])
    return -pij/evprod


def log_prob_dark_given_dark(neg_ev, amat, not_bright_slice, integration_time):
    r"""Log prob of starting in a dark state and ending in another.

    .. math::

        \mathbb{P}(s'=l'|s=l)

    :param neg_ev: The negative of the eigenvalues of the matrix :math:`M`,
        these are :math:`\omega_k` in Equation 2.25 of [Langer]_ Should be of
        shape (8, 1, 1).
    :param amat: Should be passed the result of ``akij``.
    :param not_bright_slice: A slice to pick out those components of the vector
        that are not equal to the bright state.
    :param integration_time: The (nondimensionalized) integration time.
    :return: 7x7 matrix, dimensions 0: :math:`l'` 1: :math:`l`
    """
    output, s = logsumexp(-neg_ev*integration_time, b=-amat[np.ix_(np.arange(N_STATES), not_bright_slice, not_bright_slice)], axis=-3, return_sign=True)
    if not np.allclose(np.exp(output[s < 0]), 0):
        raise ValueError("Negative values")
    return output


def log_prob_n_dark_given_dark(n_arr, neg_ev, amat, not_bright_slice, integration_time):
    r"""The log probability of getting n photons, starting in a dark state and
    ending in another dark state.

    .. math::

        \mathbb{P}(y=n, s'=l'|s=l)

    :param n_arr: The values of :math:`n` to compute for.
    :param neg_ev: The negative of the eigenvalues of the matrix :math:`M`,
        these are :math:`\omega_k` in Equation 2.25 of [Langer]_ Should be of
        shape (8, 1, 1).
    :param amat: Should be passed the result of ``akij``.
    :param not_bright_slice: A slice to pick out those components of the vector
        that are not equal to the bright state.
    :param integration_time: The (nondimensionalized) integration time.
    :return: Dimensions: 0: :math:`l'`, 1: :math:`l`, 2: :math:`n`
    """
    output = log_prob_dark_given_dark(neg_ev, amat, not_bright_slice, integration_time)
    pois = st.poisson.logpmf(n_arr, r_bg*gamma_c*integration_time)
    output = output + pois
    return output


def log_prob_n_dark_given_bright(n_arr, neg_ev, amat, not_bright_slice, integration_time):
    r"""The log probability of getting n photons, starting in the bright state
    and ending in a dark state.

    .. math::

        \mathbb{P}(y=n, s'=l'|s=\text{bright})

    :param n_arr: The values of :math:`n` to compute for.
    :param neg_ev: The negative of the eigenvalues of the matrix :math:`M`,
        these are :math:`\omega_k` in Equation 2.25 of [Langer]_ Should be of
        shape (8, 1, 1).
    :param amat: Should be passed the result of ``akij``.
    :param not_bright_slice: A slice to pick out those components of the vector
        that are not equal to the bright state.
    :param integration_time: The (nondimensionalized) integration time.
    :return: Dimensions: 0: :math:`l`, 1: :math:`n`
    """
    return np.full((len(n_arr.ravel()), N_STATES-1), -float('inf'))


def log_prob_n_bright_given_dark(n_arr, neg_ev, amat, not_bright_slice, integration_time):
    r"""The log probability of getting n photons, starting in a dark state
    and ending in the bright state.

    .. math::

        \mathbb{P}(y=n, s'=\text{bright}|s=l)

    :param n_arr: The values of :math:`n` to compute for.
    :param neg_ev: The negative of the eigenvalues of the matrix :math:`M`,
        these are :math:`\omega_k` in Equation 2.25 of [Langer]_ Should be of
        shape (8, 1, 1).
    :param amat: Should be passed the result of ``akij``.
    :param not_bright_slice: A slice to pick out those components of the vector
        that are not equal to the bright state.
    :param integration_time: The (nondimensionalized) integration time.
    :return: Dimensions: 0: :math:`l`, 1: :math:`n`
    """
    mpmath.mp.dps = 53
    output_n = []
    for nind, n in enumerate(n_arr.ravel()):
        n = mpmath.mpmathify(n.item())
        output_s = []
        for state_ind in not_bright_slice:
            output_v = 0
            for vind, v in enumerate(neg_ev.ravel()):
                v = mpmath.mpmathify(v.item())
                lower_lim = mpmath.mpmathify((gamma_c-v) * r_bg * integration_time)
                upper_lim = mpmath.mpmathify((gamma_c-v) * (1 + r_bg) * integration_time)
                upperint = reg_upper_incomp_gamma(n+1, upper_lim)
                lowerint = reg_upper_incomp_gamma(n+1, lower_lim)
                gammaintnv = -(upperint - lowerint)
                prefactornv = mpmath.exp(-v * integration_time*(1+r_bg)) * (v / gamma_c) /(1 - v/gamma_c)**(n + 1)
                output_v += amat[vind, BRIGHT_STATE, state_ind]* prefactornv*gammaintnv
            if np.isclose(float(output_v), 0):
                output_s.append(-float('inf'))
            else:
                output_s.append(float(mpmath.log(output_v)))
        output_n.append(np.array(output_s))
    return np.array(output_n)


def log_prob_bright_given_bright(neg_ev, amat, not_bright_slice, integration_time):
    r"""Log prob of starting in the bright state and ending there.

    .. math::

        \mathbb{P}(s'=\text{bright}|s=\text{bright})

    :param neg_ev: The negative of the eigenvalues of the matrix :math:`M`,
        these are :math:`\omega_k` in Equation 2.25 of [Langer]_ Should be of
        shape (8, 1, 1).
    :param amat: Should be passed the result of ``akij``.
    :param not_bright_slice: A slice to pick out those components of the vector
        that are not equal to the bright state.
    :param integration_time: The (nondimensionalized) integration time.
    :return: The log probability.
    """
    output, s = logsumexp(-neg_ev*integration_time, b=-amat[:, BRIGHT_STATE, BRIGHT_STATE], return_sign=True)
    if not np.allclose(np.exp(output[s < 0]), 0):
        raise ValueError("Negative values")
    return output


def log_prob_n_bright_given_bright(n_arr, neg_ev, amat, not_bright_slice, integration_time):
    r"""The log probability of getting n photons, starting in the bright state
    and ending in the bright state.

    .. math::

        \mathbb{P}(y=n, s'=\text{bright}|s=\text{bright})

    :param n_arr: The values of :math:`n` to compute for.
    :param neg_ev: The negative of the eigenvalues of the matrix :math:`M`,
        these are :math:`\omega_k` in Equation 2.25 of [Langer]_ Should be of
        shape (8, 1, 1).
    :param amat: Should be passed the result of ``akij``.
    :param not_bright_slice: A slice to pick out those components of the vector
        that are not equal to the bright state.
    :param integration_time: The (nondimensionalized) integration time.
    :return: Dimensions: 0: :math:`n`
    """
    output = log_prob_bright_given_bright(neg_ev, amat, not_bright_slice, integration_time)
    output = output + st.poisson.logpmf(n_arr, (1+r_bg)*gamma_c*integration_time)
    return output


def log_prob_n_l_given_lp(n, integration_time):
    r"""Log prob of getting :math:`n` photons, starting in state :math:`l` and
    ending in state :math:`l'`.

    .. math::

        \mathbb{P}(y=n, s'=l'|s=l)

    :param n: An array of number of photons to compute for.
    :param integration_time: The (nondimensionalized) integration time to
        compute the probabilities for.
    :return: Dimensions: 0: :math:`n`, 1: :math:`l'`, 2: :math:`l`
    """
    m = m_matrix()
    n_arr = np.array(n)
    n_shape = n_arr.shape
    n_arr = n_arr.reshape((-1,))
    output_dist = np.zeros((len(n_arr), N_STATES, N_STATES))
    ev = np.linalg.eigvals(m)
    amat = akij(m, ev)
    neg_ev = -ev
    # HACK: Very small eigenvalues. Need to take a log later, so we enforce
    # positivity, introduces some numerical error.
    neg_ev[np.isclose(neg_ev, 0)] = np.abs(neg_ev[np.isclose(neg_ev, 0)])
    not_bright_slice = np.arange(N_STATES)
    not_bright_slice = np.concatenate([not_bright_slice[:BRIGHT_STATE], not_bright_slice[BRIGHT_STATE+1:]])

    output_dist[:, BRIGHT_STATE, BRIGHT_STATE] = log_prob_n_bright_given_bright(n_arr, neg_ev, amat, not_bright_slice, integration_time)

    neg_ev = neg_ev[:, None]
    n_arr = n_arr[:, None, None]

    output_dist[:, BRIGHT_STATE, not_bright_slice] = log_prob_n_bright_given_dark(n_arr, neg_ev, amat, not_bright_slice, integration_time)
    output_dist[:, not_bright_slice, BRIGHT_STATE] = log_prob_n_dark_given_bright(n_arr, neg_ev, amat, not_bright_slice, integration_time)

    neg_ev = neg_ev[:, :, None]

    output_dist[np.ix_(np.arange(len(n_arr)), not_bright_slice, not_bright_slice)] = log_prob_n_dark_given_dark(n_arr, neg_ev, amat, not_bright_slice, integration_time)

    output_dist = output_dist.reshape(n_shape + (N_STATES, N_STATES))
    return output_dist


def log_prob_l_zero_given_lp(integration_time):
    r"""Log prob of seeing no counts, starting in state :math:`l'` and ending
    in state :math:`l`.

    .. math::

        \mathbb{P}(s'=l, y=0|s=l')

    :param integration_time: The (nondimensionalized) integration time
    :return: Dimensions: 0: :math:`l'`, 1: :math:`l`
    """
    return log_prob_n_l_given_lp(np.array(0, dtype=int), integration_time).transpose()


def log_prob_l_nonzero_given_lp(integration_time):
    r"""Log prob of seeing nonzero counts, starting in state :math:`l'` and
    ending in state :math:`l`.

    .. math::

        \mathbb{P}(s'=l, y\neq 0|s=l')

    :param integration_time: The (nondimensionalized) integration time
    :return: Dimensions: 0: :math:`l'`, 1: :math:`l`
    """
    retval = log_prob_l_zero_given_lp(integration_time)
    tm = log_transition_matrix(integration_time)
    retval, signs = logsumexp(np.stack([tm, retval], axis=-1), axis=-1, b=np.array([1., -1.]), return_sign=True)
    if not np.allclose(np.exp(retval[signs < 0]), 0):
        raise ValueError("Negative values in transition matrix.")
    return retval


def log_prob_n_given_l(n, integration_time):
    r"""Log prob of getting :math:`n` photons, starting in state :math:`l`.

    .. math::

        \mathbb{P}(y=n|s=l)

    :param n: The array of photon counts to compute probailities for.
    :param integration_time: The (nondimensionalized) integration time
    :return: Dimensions: 0: :math:`n`, 1: :math:`l`
    """
    output = log_prob_n_l_given_lp(n, integration_time)
    return logsumexp(output, axis=-2)


def log_transition_matrix(integration_time):
    r"""Log of the transition probability.

    .. math::

        \mathbb{P}(s'=l'|s=l)

    :param integration_time: The (nondimensionalized) integration time
    :return: Dimensions: 0: :math:`l`, 1: :math:`l'`
    """
    m = m_matrix()
    ev = np.linalg.eigvals(m)
    amat = akij(m, ev)
    ev = ev[:, None, None]
    retval, signs = logsumexp(ev*integration_time, axis=-3, b=-amat, return_sign=True)
    if not np.allclose(np.exp(retval[signs < 0]), 0):
        raise ValueError("Negative values in transition matrix.")
    return retval.transpose()


def expanded_transitions(integration_time, k=2):
    if k < 2:
        raise ValueError("Need to have at least two outcomes.")
    n = np.arange(k-1)
    nllp = log_prob_n_l_given_lp(n, integration_time)
    tm = log_transition_matrix(integration_time)
    tm = np.transpose(tm)
    total_llp = logsumexp(nllp, 0)
    remain, signs = logsumexp(np.stack([tm, total_llp], axis=-1), axis=-1, b=np.array([1., -1.]), return_sign=True)
    assert np.allclose(np.exp(remain[signs < 0]), 0)
    expanded_tm = np.concatenate([nllp, remain[None, ...]], axis=0)
    expanded_tm = np.tile(expanded_tm, (k, 1, 1, 1))
    expanded_tm = np.moveaxis(expanded_tm, (0,1,2,3), (1, 3, 2, 0))
    expanded_tm = np.reshape(expanded_tm, (k*N_STATES, k*N_STATES))
    return expanded_tm


def expanded_permutations(perms, k=2):
    if k < 2:
        raise ValueError("Need to have at least two outcomes.")
    retval = np.moveaxis(np.tile(perms, (k, 1, 1)), (0, 1, 2), (2, 0, 1))
    retval *= k
    for i in range(k):
        retval[..., i] += i
    retval = retval.reshape((-1, k*N_STATES))
    return retval


def expanded_outcomes(k=2):
    if k < 2:
        raise ValueError("Need to have at least two outcomes.")
    retval = np.log(np.eye(k) + ZERO)
    retval -= logsumexp(retval, axis=-1)
    retval = np.tile(retval, (N_STATES, 1, 1))
    retval = np.reshape(retval, (k*N_STATES, k))
    return retval


def expanded_initial(k=2):
    retval = np.full((k*N_STATES,), np.log(ZERO))
    retval[lo_to_i((BRIGHT_STATE, 0), k=k)] = np.log(.5)
    retval[lo_to_i((DARK_STATE, 0), k=k)] = np.log(.5)
    retval -= logsumexp(retval, -1)
    return retval


def lo_to_i(lo, k=2):
    return k*lo[0] + lo[1]


def i_to_lo(i, k=2):
    return divmod(i, k)


def hyperfine_graph():
    r"""The graph whose vertices are levels, and the edges are those that differ
    by :math:`\Delta F = 1` and :math:`\Delta m_F = 0, \pm 1`

    :return: A :py:class:`~networkx.Graph` object describing the graph.
    """
    G = nx.Graph()
    G.add_nodes_from(l_to_fmf.values())
    for node1 in G.nodes:
        for node2 in G.nodes:
            if (node1 != node2) and (node1[0] != node2[0]) and (
                    abs(node1[1] - node2[1]) <= 1):
                G.add_edge(node1, node2)
    return G


def path_to_perm(path):
    r"""Given a path on the graph, returns a permutation on the corresponding
    states.

    :param path: A list of states :math:`(F, m_F)`
    :return: Shape (8,)
    """
    perm = np.arange(N_STATES, dtype=int)
    revpath = list(reversed(path))
    for i in range(len(revpath) - 1):
        perm[fmf_to_l[revpath[i]]] = fmf_to_l[revpath[i + 1]]
    perm[fmf_to_l[revpath[-1]]] = fmf_to_l[revpath[0]]
    return perm


def allowable_permutations():
    r"""Gets all shortest paths between distinct states on ``hyperfine_graph``.

    :return: Shape ``(8*7/2 + 1, 8)``
    """
    G = hyperfine_graph()
    paths = []
    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 != node2:
                paths.append(nx.shortest_path(G, node1, node2))
    perms = np.stack([path_to_perm(path) for path in paths], axis=0)
    perms = np.concatenate([np.arange(N_STATES, dtype=int)[None, :], perms], axis=0)
    return perms


def switch_with_bright():
    r"""Only those swaps with bright along paths of ``hyperfine_graph``, and
    their inverses.

    :return: Shape ``(2*7 + 1, 8)``
    """
    G = hyperfine_graph()
    paths = []
    node1 = l_to_fmf[BRIGHT_STATE]
    for node2 in G.nodes:
        if node1 != node2:
            paths.append(nx.shortest_path(G, node1, node2))
    perms = np.stack([path_to_perm(path) for path in paths], axis=0)
    invperms = []
    for perm in perms:
        invperms.append(np.argsort(perm))
    perms = np.array(list(set([tuple(perm) for perm in (list(perms) + invperms)])))
    perms = np.concatenate([np.arange(N_STATES, dtype=int)[None, :], perms], axis=0)
    return perms


def next_simplest_perms():
    G = hyperfine_graph()
    path = nx.shortest_path(G, l_to_fmf[BRIGHT_STATE], l_to_fmf[DARK_STATE])
    perm = path_to_perm(path)
    invperm = np.argsort(perm)
    neighboring_path = nx.shortest_path(G, l_to_fmf[DARK_STATE], (1, 0))
    neighboring_perm = path_to_perm(neighboring_path)
    neighboring_inv = np.argsort(neighboring_perm)
    other_path = nx.shortest_path(G, l_to_fmf[DARK_STATE], (2, 0))
    other_perm = path_to_perm(other_path)
    perms = np.concatenate([
        np.arange(N_STATES, dtype=int)[None, :],
        perm[None, :],
        invperm[None, :],
        neighboring_perm[None, :],
        neighboring_inv[None, :],
        other_perm[None, :],
    ], axis=0)
    return perms


def simplest_perms():
    r"""Only the swap :math:`(1, -1) \leftrightarrow (2, 2)`, its inverse, and
    the identity.

    :return: Shape ``(3, 8)``
    """
    G = hyperfine_graph()
    path = nx.shortest_path(G, l_to_fmf[BRIGHT_STATE], l_to_fmf[DARK_STATE])
    perm = path_to_perm(path)
    invperm = np.argsort(perm)
    perms = np.concatenate([np.arange(N_STATES, dtype=int)[None, :], perm[None, :], invperm[None, :]], axis=0)
    return perms


def plot_bright_pop():
    times = np.arange(1e-6, 2e-4, 1e-6)
    to_plot = np.empty((len(times), N_STATES-1))
    not_bright_slice = np.arange(N_STATES)
    not_bright_slice = np.concatenate([not_bright_slice[:BRIGHT_STATE], not_bright_slice[BRIGHT_STATE+1:]])
    for i, integ in enumerate(np.arange(1e-6, 2e-4, 1e-6)):
        to_plot[i] = np.exp(log_transition_matrix(integ * dimensionful_gamma)[not_bright_slice, BRIGHT_STATE])
    labels = [l_to_fmf[l] for l in not_bright_slice]
    plt.plot(times*1e6, to_plot, label=labels)
    plt.ylim([0, .3])
    plt.legend()
    plt.show()



def plot_dark_bright_histograms():
    all_histograms = log_prob_n_given_l(np.arange(70), 4e-4*dimensionful_gamma)
    plt.bar(np.arange(40), np.exp(all_histograms[:40, DARK_STATE]), log=True)
    plt.show()
    plt.bar(np.arange(70), np.exp(all_histograms[:, BRIGHT_STATE]), log=True)
    plt.show()


def unbinned_hists(integration_time, max_photons=10):
    hists = log_prob_n_given_l(np.arange(max_photons), integration_time)
    total_weight = logsumexp(hists, -2)
    hists[-1] = logsumexp(np.stack([hists[-1, :], log1mexp(total_weight)], axis=-2), axis=-2)
    hists = hists[..., [DARK_STATE, BRIGHT_STATE]]
    hists = hists.transpose()
    return hists


def main(args):
    integration_time = args.integration_time * dimensionful_gamma
    filename = args.filename
    if filename.split(".")[-1] != "npz":
        filename += ".npz"

    plot_bright_pop()
    plot_dark_bright_histograms()
    initial = expanded_initial()
    transition = expanded_transitions(integration_time)
    outcome = expanded_outcomes()
    perms = expanded_permutations(allowable_permutations())

    np.savez(
        filename,
        initial=initial,
        transition=transition,
        outcome=outcome,
        perms=perms,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Get parameter matrices for Beryllium")
    parser.add_argument(
        "-o",
        "--filename",
        help=".npz filename to write to. Extension will be added if not already .npz,"
             " extension will be appended.",
        type=str,
    )
    parser.add_argument(
        "integration_time",
        metavar="integration-time",
        help="The amount of time to integrate the collection of photons for,"
             "in units of 1/(2 * pi * 19.4 * 10 ** 6 Hz)",
        type=float,
    )
    args = parser.parse_args()
    main(args)
