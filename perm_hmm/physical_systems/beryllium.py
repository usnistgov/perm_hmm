"""
Computes the transition matrix and output probabilities of the Beryllium ion.
All equations from [Langer]_, Chapter 2.

.. [Langer] C. E. Langer, High fidelity quantum information processing with
     trapped ions, Ph.D. thesis (2006), Last updated - 2016-05-25.
"""
import numpy as np
import scipy.stats
import sys
import scipy
import scipy.special
import networkx as nx
from perm_hmm.util import ZERO


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
BRIGHT_STATE = fmf_to_l[(2, 2)]
DARK_STATE = fmf_to_l[(1, -1)]

dimensionful_gamma = 2 * np.pi * 19.4 * 10 ** 6
dimensionful_delta_z = 233 * 10 ** 6
dimensionful_detection_laser = 2 * np.pi * 957200 * 10 ** 9
dimensionful_A = -625.0088370 * 10 ** 6
dimensionful_gamma_c = 30 / (330 * 10 ** (-6))
epsilon_sigma_minus = 1 * 10 ** (-3)

gi_prime = 2.13477985 * 10 ** (-4)
s0 = 1 / 2
delta_z = dimensionful_delta_z / dimensionful_gamma
A = dimensionful_A / dimensionful_gamma
gamma = dimensionful_gamma / dimensionful_gamma
gamma_prime = gamma / 2 * np.sqrt(1 + s0)
g_squared = gamma ** 2 * s0 / 8

dimensionful_bohr_magneton = 9.27401008 * 10 ** (-24)
dimensionful_magnetic_field = .0119
dimensionful_planck = 6.626 * 10 ** (-34)
gj = -2.00226206

x0 = -dimensionful_bohr_magneton * dimensionful_magnetic_field * gj * (
        1 - gi_prime) / (dimensionful_planck * dimensionful_A)
f_plus = 2
gamma_c = dimensionful_gamma_c / dimensionful_gamma
gamma_bg = .06 * gamma_c



def energy(f, mf, x):
    if f == 2:
        plus_or_minus = 1
    elif f == 1:
        plus_or_minus = -1
    return (
        2 * np.pi * A *
        (-1 / 4 + gi_prime / (1 - gi_prime) * mf * x + plus_or_minus * np.sqrt(
            f_plus ** 2 + 2 * mf * x + x ** 2
        ))
    )


def plus_prefactor(f, mf, x):
    if f == 2:
        plus_or_minus = 1
    elif f == 1:
        plus_or_minus = -1
    return (
        (mf + x + plus_or_minus * np.sqrt(
            f_plus ** 2 + 2 * mf * x + x ** 2
        )) / np.sqrt(f_plus ** 2 - mf ** 2)
    )


def alpha(l, x):
    f, mf = l_to_fmf[l]
    if mf == -f_plus:
        return 1
    if mf == f_plus:
        return 0
    return (
        1 / np.sqrt(1 + plus_prefactor(f, mf, x) ** 2)
    )


def beta(l, x):
    f, mf = l_to_fmf[l]
    if mf == -f_plus:
        return 0
    if mf == f_plus:
        return 1
    return (
        plus_prefactor(f, mf, x) / np.sqrt(
            1 + plus_prefactor(f, mf, x) ** 2
        )
    )


def delta_3_over_2(l, x):
    f, mf = l_to_fmf[l]
    return np.abs(energy(f, mf, x) - energy(2, 2, x))


def gamma_deltam_0(i, j, x):
    if i == j:
        return 0
    delta_mf = l_to_fmf[i][1] - l_to_fmf[j][1]
    if delta_mf != 0:
        return 0
    return (
        g_squared * gamma * 4 / 9 * np.abs(alpha(j, x) * beta(j, x)) ** 2 *
        ((delta_3_over_2(j, x) + 3 / 2 * delta_z) ** 2 + gamma_prime ** 2) /
        ((delta_3_over_2(j, x) ** 2 + delta_3_over_2(j, x) *
         delta_z - gamma_prime ** 2) ** 2 +
         gamma ** 2 * (2 * delta_3_over_2(j, x) + delta_z) ** 2)
    )


def gamma_deltam_1(i, j, x):
    delta_mf = l_to_fmf[i][1] - l_to_fmf[j][1]
    if delta_mf == 1:
        return (
                g_squared * gamma * 2 / 9 *
                np.abs(alpha(j, x) * beta(i, x)) ** 2 /
                (np.abs(delta_3_over_2(j, x) + 1j * gamma_prime + delta_z) ** 2)
        )
    return 0


def bf(j, t):
    if j == fmf_to_l[(2, 2)]:
        return 1 - np.exp(-(gamma_c + gamma_bg) * t)
    else:
        return 1 - np.exp(-gamma_bg * t)

def adjugate(m):
    n = m.shape[-1]
    if m.shape[-2:] != (n, n):
        raise ValueError("Can't take the adjugate of a non-square matrix")
    minor_slicer = np.tile(np.arange(n), (n, 1))
    minor_slicer = np.tril(minor_slicer, -1)[:, :-1] +  np.triu(minor_slicer, 1)[:, 1:]
    slicer = (np.moveaxis(minor_slicer[:, :, None, None], [1, 2], [2, 1]), np.moveaxis(minor_slicer[None, None, :, :], [1, 2], [2, 1]))
    signs = -((((np.arange(n)[:, None] + np.arange(n)[None, :]) % 2) * 2) - 1)
    return np.moveaxis(np.linalg.det(m[(..., ) + slicer]), [-2, -1], [-1, -2])*signs


def m_matrix():
    gamma_0_x0 = np.array(
        [[gamma_deltam_0(i, j, x0) for j in range(8)] for i in range(8)])
    gamma_1_x0 = np.array(
        [[gamma_deltam_1(i, j, x0) for j in range(8)] for i in range(8)])

    gamma_minus_x0 = np.array(
        [[epsilon_sigma_minus * gamma_deltam_1(j, i, x0) for j in range(8)] for
         i in range(8)])
    gamma_minus_x0 += np.array(
        [[epsilon_sigma_minus * gamma_deltam_0(j, i, x0) for j in range(8)] for
         i in range(8)])

    gamma_x0 = gamma_0_x0 + gamma_1_x0 + gamma_minus_x0

    m_x0 = gamma_x0 - np.diagflat(np.ones((1, 8)).dot(gamma_x0))
    return m_x0


def polymat(m, s):
    m_shape = m.shape
    arr_s = np.array(s)
    s_shape = arr_s.shape
    reshaped_s = arr_s.reshape((-1, 1, 1))
    n = m.shape[-1]
    eye = np.eye(n)
    val = adjugate(m[..., None, :, :] - eye*reshaped_s).reshape(m_shape[:-2] + s_shape + (n, n))
    return val


def akij(m, ev=None):
    n = m.shape[-1]
    if ev is None:
        ev = np.linalg.eigvals(m)
    pij = adjugate(m[..., None, :, :] - np.eye(m.shape[-1])*ev[..., None, None])
    evprod = np.expand_dims(np.prod(ev[..., None, :] - ev[..., :, None] + np.eye(n), axis=-1), [-1, -2])
    return -pij/evprod


def prob_of_n_photons(n, dimensionful_integration_time):
    integration_time = dimensionful_integration_time*dimensionful_gamma
    m = m_matrix()
    num_states = m.shape[-1]
    r_bg = gamma_bg/gamma_c
    n_arr = np.array(n)
    n_shape = n.shape
    n_arr = n_arr.reshape((-1, 1, 1))
    output_dist = np.zeros((len(n_arr), num_states))
    ev = np.linalg.eigvals(m)
    amat = akij(m, ev)
    neg_ev = -ev
    neg_ev = neg_ev[None, :, None]
    not_bright_slice = np.arange(num_states)
    not_bright_slice = np.concatenate([not_bright_slice[:BRIGHT_STATE], not_bright_slice[BRIGHT_STATE+1:]])
    dark_dist = (
        1 + (amat[:, BRIGHT_STATE, not_bright_slice] * np.exp(-neg_ev * integration_time)).sum(-2)
        )*scipy.stats.poisson.pmf(np.squeeze(n_arr, -1), integration_time * r_bg * gamma_c) + \
        (
            amat[:, BRIGHT_STATE, not_bright_slice] * np.exp(-neg_ev * integration_time*(1+r_bg)) * (
                np.exp((np.log(neg_ev) - np.log(gamma_c)) - np.log(-np.expm1(np.log(neg_ev)-np.log(gamma_c))) * (n_arr + 1)) *
                # (neg_ev * gamma_c ** n_arr) / (gamma_c - neg_ev) ** (n_arr + 1) *
                (
                    scipy.special.gammainc(n_arr + 1, (gamma_c-neg_ev) * (1 + r_bg) * integration_time) -
                    scipy.special.gammainc(n_arr + 1, (gamma_c-neg_ev) * r_bg * integration_time)
                )
            )
        ).sum(axis=-2)
    n_arr = n_arr.squeeze(-1)
    neg_ev = neg_ev.squeeze(-1)
    bright_dist = (
        amat[:, BRIGHT_STATE, BRIGHT_STATE] * (
            -np.exp(-neg_ev*integration_time) *
            scipy.stats.poisson.pmf(n_arr, (1 + r_bg) * gamma_c * integration_time) -
            np.exp((np.log(neg_ev) - np.log(gamma_c) + neg_ev * integration_time * r_bg) - np.log1p(np.exp(np.log(neg_ev) - np.log(gamma_c))) * (n_arr + 1)) *
            # (neg_ev * gamma_c ** n_arr * np.exp(neg_ev * integration_time * r_bg)) / (gamma_c + neg_ev) ** (n_arr + 1) *
            (
                scipy.special.gammainc(n_arr + 1, (gamma_c+neg_ev) * (1 + r_bg) * integration_time) -
                scipy.special.gammainc(n_arr + 1, (gamma_c+neg_ev) * r_bg * integration_time)
            )
        )
    ).sum(-1)
    output_dist[:, not_bright_slice] = dark_dist
    output_dist[:, BRIGHT_STATE] = bright_dist
    output_dist[np.abs(output_dist) < ZERO] = 0.
    return output_dist.reshape(n_shape + (num_states,))


def transition_matrix(dimensionful_integration_time):
    m = m_matrix()
    ev = np.linalg.eigvals(m)
    amat = akij(m, ev)
    ev = ev[:, None, None]
    integration_time = dimensionful_integration_time * dimensionful_gamma
    pij = (-np.exp(ev*integration_time)*amat).sum(-3).transpose()
    pij[np.abs(pij) < ZERO] = ZERO
    log_pij = np.log(pij)
    log_pij -= scipy.special.logsumexp(log_pij, axis=-1, keepdims=True)
    return np.exp(log_pij)


def bernoulli_parameters(dimensionful_integration_time):
    pij = transition_matrix(dimensionful_integration_time)
    bright_probs = 1-prob_of_n_photons(np.array(0, dtype=int), dimensionful_integration_time)
    bright_or_dark = np.log([ZERO, ZERO, ZERO, ZERO, 1 / 2, ZERO, ZERO, 1 / 2])
    bright_or_dark -= scipy.special.logsumexp(bright_or_dark)
    return bright_or_dark, pij, bright_probs


def old_parameter_matrices(dimensionful_integration_time=11 * 10 ** (-6)):
    integration_time = dimensionful_integration_time * dimensionful_gamma
    gamma_0_x0 = np.array(
        [[gamma_deltam_0(i, j, x0) for j in range(8)] for i in range(8)])
    gamma_1_x0 = np.array(
        [[gamma_deltam_1(i, j, x0) for j in range(8)] for i in range(8)])

    gamma_minus_x0 = np.array(
        [[epsilon_sigma_minus * gamma_deltam_1(j, i, x0) for j in range(8)] for
         i in range(8)])
    gamma_minus_x0 += np.array(
        [[epsilon_sigma_minus * gamma_deltam_0(j, i, x0) for j in range(8)] for
         i in range(8)])

    gamma_x0 = gamma_0_x0 + gamma_1_x0 + gamma_minus_x0

    m_x0 = gamma_x0 - np.diagflat(np.ones((1, 8)).dot(gamma_x0))

    eigenValues, eigenVectors = np.linalg.eig(m_x0)

    idx = eigenValues.argsort()[::-1]
    eigenValues = eigenValues[idx]
    eigenVectors = eigenVectors[:, idx]

    u = eigenVectors
    inv = np.linalg.inv(u)

    output_means = np.array(
        [integration_time * (gamma_c + gamma_bg) if j == fmf_to_l[(2, 2)]
            else integration_time * gamma_bg for j in range(8)],
        dtype=np.float32,
    )
    pij = np.array(
        [u.dot(inv.dot(v0) * np.exp(eigenValues * integration_time))
            for v0 in np.eye(8)],
        dtype=np.float32,
    )
    pij[pij < ZERO] = ZERO
    pij /= pij.sum(axis=1, keepdims=True)

    return output_means, pij


def old_bernoulli_parameters(dimensionful_integration_time=11 * 10 ** (-6)):
    output_means, pij = old_parameter_matrices(dimensionful_integration_time)
    bright_probs = 1 - scipy.stats.poisson.pmf(0, mu=output_means)
    bright_or_dark = np.log([ZERO, ZERO, ZERO, ZERO, 1 / 2, ZERO, ZERO, 1 / 2])
    bright_or_dark -= scipy.special.logsumexp(bright_or_dark)
    return bright_or_dark, pij, bright_probs


def allowable_permutations():
    G = nx.Graph()
    G.add_nodes_from(l_to_fmf.values())
    for node1 in G.nodes:
        for node2 in G.nodes:
            if (node1 != node2) and (node1[0] != node2[0]) and (
                    abs(node1[1] - node2[1]) <= 1):
                G.add_edge(node1, node2)
    paths = []
    for node1 in G.nodes:
        for node2 in G.nodes:
            if node1 != node2:
                paths.append(nx.shortest_path(G, node1, node2))
    perms = np.tile(np.arange(8, dtype=int), (len(paths) + 1, 1))
    for j in range(len(paths)):
        revpath = list(reversed(paths[j]))
        for i in range(len(revpath) - 1):
            perms[j, fmf_to_l[revpath[i]]] = fmf_to_l[revpath[i + 1]]
        perms[j, fmf_to_l[revpath[-1]]] = fmf_to_l[revpath[0]]
    return perms
