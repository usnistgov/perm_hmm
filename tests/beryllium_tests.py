import pytest
from perm_hmm.util import ZERO
import example_systems.beryllium as beryllium
from example_systems.beryllium import N_STATES, BRIGHT_STATE, DARK_STATE
import numpy as np
from scipy.special import logsumexp, logit, expit
import itertools


def expanded_transitions(integration_time):
    r"""Log transition matrix on the expanded state space.

    .. math::

        \mathbb{P}(s'=l, y=o|s=l', y_{\text{previous}}=o')

    :param integration_time: The (nondimensionalized) integration time
    :return: Dimensions: 0: The joint index for :math:`(s, y_{\text{previous}})`
        1: The joint index for :math:`(s', y)`
        Use ``lo_to_i`` to convert indices.
    """
    lzl = beryllium.log_prob_l_zero_given_lp(integration_time)
    lnl = beryllium.log_prob_l_nonzero_given_lp(integration_time)
    total = np.moveaxis(np.tile(np.stack((lzl, lnl), axis=1), (2, 1, 1, 1)), (0, 1, 2, 3), (1, 0, 3, 2))
    return total.reshape((16, 16))


def i_to_lo(i):
    r"""To convert indices of the output of ``expanded_transitions`` into
    :math:`(l, o)` pairs.

    :param i: The index.
    :return: divmod(i, 2)
    """
    return divmod(i, 2)


def lo_to_i(lo):
    r"""To convert pairs :math:`(l, o)` into indices for the output of
    ``expanded_transitions``

    :param lo: The (l, o) pair
    :return: 2*lo[0] + lo[1]
    """
    return 2*lo[0] + lo[1]


def expanded_outcome_logits():
    r"""Log odds of the outcomes on the expanded state space.

    .. math::

        \mathbb{P}(y=o|s=l, y=o') = \delta_{o, o'}

    When we tatke the log-odds of this, we get plus and minus infinity. These
    values are clipped to ``scipy.logit(beryllium.ZERO)`` and its negative.

    :return: np.tile([logit(ZERO), -logit(ZERO)], 8)
    """
    return np.tile([logit(ZERO), -logit(ZERO)], 8)


def expanded_permutations(perms):
    r"""Given permutations on the unexpanded state space, returns corresponding
    permutations on the expanded state space.

    :param perms: Shape ``(n_perms, 8)``.
    :return: Shape ``(n_perms, 16)``.
    """
    retval = np.moveaxis(np.tile(perms, (2, 1, 1)), (0, 1, 2), (2, 0, 1))
    retval[..., 0] *= 2
    retval[..., 1] *= 2
    retval[..., 1] += 1
    retval = retval.reshape((-1, 2*N_STATES))
    return retval


def expanded_initial():
    r"""The initial state distribution on the expanded state space.

    Assumes a uniform prior on :math:`(F, m_F) = (2, 2)` and :math:`(1, -1)`.

    :return: Shape (16,)
    """
    retval = np.full((16,), np.log(ZERO))
    retval[lo_to_i((BRIGHT_STATE, 0))] = np.log(.5)
    retval[lo_to_i((DARK_STATE, 0))] = np.log(.5)
    retval -= logsumexp(retval, -1)
    return retval


def test_adjugate():
    for n in range(2, 10):
        m = np.random.rand(n, n)
        adj = beryllium.adjugate(m)
        assert (np.allclose(np.matmul(m, adj), np.linalg.det(m)*np.eye(n)))
        m = np.random.rand(1, n, n)
        adj = beryllium.adjugate(m)
        assert (np.allclose(np.matmul(m, adj), np.linalg.det(m)[..., None, None]*np.eye(n)))
        m = np.random.rand(3, 4, n, n)
        adj = beryllium.adjugate(m)
        assert (np.allclose(np.matmul(m, adj), np.linalg.det(m)[..., None, None]*np.eye(n)))


@pytest.mark.parametrize('n', list(range(2, 10)))
def test_polymat(n):
    m = np.random.rand(n, n)
    ev = np.linalg.eigvals(m)
    adj_resolvent = beryllium.polymat(m, ev)
    assert (
        np.allclose(np.matmul(
            m-np.eye(n)*ev[..., :, None, None],
            adj_resolvent
        ), 0)
    )


@pytest.mark.parametrize('n', list(range(2, 5)))
def test_akij(n):
    mlist = [np.random.rand(n, n), np.random.rand(1, n, n), np.random.rand(3, 4, n, n)]
    slist = [np.random.randn(), np.random.randn(1), np.random.randn(7, 6)]
    for m, s in itertools.product(mlist, slist):
        s = np.array(s)
        pij = beryllium.polymat(m, s)
        if s.shape == ():
            d = np.linalg.det(m - np.eye(n)*s)
        else:
            d = np.linalg.det(m.reshape(m.shape[:-2]+(1,)*len(s.shape)+m.shape[-2:]) - np.eye(n)*s[..., None, None])
        resolvent_1 = pij/d[..., None, None]
        ev = np.linalg.eigvals(m)
        akij_with_ev = beryllium.akij(m, ev)
        akij_no_ev = beryllium.akij(m)
        assert (np.allclose(akij_with_ev, akij_no_ev))
        if s.shape == ():
            resolvent_2 = (akij_no_ev/(s-ev[..., :, None, None])).sum(-3)
            resolvent_3 = np.linalg.inv(m-np.eye(n)*s)
        else:
            resolvent_2 = (akij_no_ev[(...,) + (None,)*len(s.shape) + (slice(None),)*3]/(s[..., None, None, None]-ev[(...,)+(None,)*len(s.shape) + (slice(None), None, None)])).sum(-3)
            resolvent_3 = np.linalg.inv(m[(...,) + (None,)*len(s.shape)+(slice(None),)*2]-np.eye(n)*s[..., None, None])
        assert (np.allclose(resolvent_1, resolvent_2))
        assert (np.allclose(resolvent_1, resolvent_3))
        assert (np.allclose(resolvent_3, resolvent_2))


@pytest.mark.parametrize('time', np.arange(-9, -3, 1))
def test_output_dist(time):
    n_array = np.arange(3000)
    time = np.exp(time)
    integration_time = time * beryllium.dimensionful_gamma
    output_dist = beryllium.log_prob_n_l_given_lp(n_array, integration_time)
    assert (np.all(output_dist < 0.))
    transition_matrix = logsumexp(output_dist, axis=-3)
    ltm = beryllium.log_transition_matrix(integration_time)
    diff = transition_matrix - ltm.transpose()
    assert (np.allclose(transition_matrix, ltm.transpose()))
    total_norm = logsumexp(transition_matrix, axis=-2)
    assert (np.allclose(total_norm, 0.))


@pytest.mark.parametrize('time', np.arange(-9, -3, 1))
def test_expanded_matrix(time):
    time = np.exp(time)
    integration_time = time * beryllium.dimensionful_gamma
    bt = beryllium.expanded_transitions(integration_time)
    tm = expanded_transitions(integration_time)
    assert np.allclose(np.exp(bt), np.exp(tm))


def test_expanded_initial():
    il = expanded_initial()
    bil = beryllium.expanded_initial()
    assert np.allclose(np.exp(bil), np.exp(il))


def test_expanded_outcomes():
    ol = expanded_outcome_logits()
    bol = beryllium.expanded_outcomes()
    assert np.allclose(np.exp(bol)[:, 1], expit(ol))


def test_expanded_perms():
    p = beryllium.allowable_permutations()
    ep = expanded_permutations(p)
    bep = beryllium.expanded_permutations(p)
    assert np.all(ep == bep)
