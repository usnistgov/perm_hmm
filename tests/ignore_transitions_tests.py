import pytest

import numpy as np
from scipy.special import logsumexp, log1p
import torch
import pyro.distributions as dist

from perm_hmm.util import num_to_data
from perm_hmm.policies.ignore_transitions import IgnoreTransitions
from perm_hmm.models.hmms import PermutedDiscreteHMM
from perm_hmm.classifiers.perm_classifier import PermClassifier
from perm_hmm.postprocessing import ExactPostprocessor
from perm_hmm.policies.min_tree import MinEntPolicy
from perm_hmm.rate_comparisons import exact_rates
from adapt_hypo_test.two_states import util, no_transitions as nt
from adapt_hypo_test.two_states.util import nx_to_log_probs, pq_to_m, m_to_r




@pytest.mark.parametrize('p, q, n', [
    (.1, .2, 5),
])
def test_ignore_transitions(p, q, n):
    it = IgnoreTransitions(torch.eye(2, dtype=int), p, q, 0, 1)
    sigmas, chi = nt.solve(p, q, n)
    it.solve(n)
    assert all([np.all(s == sp) for s, sp in zip(sigmas, it.sigmas)])
    yn = np.stack([num_to_data(num, n) for num in range(2**n)])
    x, applied = nt.evaluate_sigmas(sigmas, yn)
    it_applied = it.get_perms(torch.from_numpy(yn))
    bool_it_applied = (it_applied == torch.tensor([1, 0], dtype=int).unsqueeze(-2).unsqueeze(-2)).all(-1)
    assert np.all(applied[..., 1:] == bool_it_applied.numpy()[..., :-1])


@pytest.mark.parametrize('p, q, n', [
    (.1, .2, 5),
])
def test_it_rate(p, q, n):
    il = (torch.ones(2)/2).log()
    tl = (torch.eye(2) + np.finfo(float).eps).log()
    tl -= tl.logsumexp(-1, keepdim=True)
    observation_dist = dist.Bernoulli(torch.tensor([p, 1-q]))
    hmm = PermutedDiscreteHMM(il, tl, observation_dist)
    cf = PermClassifier(hmm)
    yn = np.stack([num_to_data(num, n) for num in range(2**n)])
    tyn = torch.from_numpy(yn)
    sigmas, chi = nt.solve(p, q, n)
    x, applied = nt.evaluate_sigmas(sigmas, yn)
    perms = torch.tensor([0, 1], dtype=int).expand(yn.shape + (2,)).clone().detach()
    perms[applied] = torch.tensor([1, 0], dtype=int)
    perms = torch.roll(perms, -1, dims=-2)
    c, d = cf.classify(tyn, perms=perms, verbosity=2)
    log_joint = hmm.posterior_log_initial_state_dist(tyn, perms).T + hmm.log_prob(tyn, perms)
    ntlj = nt.log_joint(yn.astype(int), util.pq_to_m(p, q), sigmas)
    assert np.allclose(ntlj, log_joint.numpy())
    lp = hmm.log_prob(tyn, perms)
    ntlp = nt.lp(yn.astype(int), util.pq_to_m(p, q), sigmas)
    assert np.allclose(ntlp, lp)

    ep = ExactPostprocessor(log_joint, c)
    r = ep.log_misclassification_rate()
    rp = log1p(-np.exp(logsumexp(chi.ravel(), axis=-1) - np.log(2)))
    assert np.isclose(r.numpy(), rp)


def pq_to_hmm(p, q):
    tl = (torch.eye(2) + np.finfo(float).eps).log()
    tl -= tl.logsumexp(-1, keepdim=True)
    il = torch.ones(2) - torch.log(torch.tensor(2.))
    observation_dist = dist.Bernoulli(torch.tensor([p, 1-q]))
    return PermutedDiscreteHMM(il, tl, observation_dist)


def two_transitions():
    return torch.eye(2, dtype=int)


def me_policy(hmm):
    return MinEntPolicy(two_transitions(), hmm, save_history=True)


def nt_policy(p, q):
    return IgnoreTransitions(two_transitions(), p, q, 0, 1, save_history=True)


@pytest.mark.parametrize("p, q, n", [
    (.1, .2, 10),
    (.3, .5, 8),
    (.03, .8, 8),
])
def test_rates(p, q, n):
    hmm = pq_to_hmm(p, q)
    nt_s = nt_policy(p, q)
    nt_s.solve(n)
    all_data = np.stack([num_to_data(num, n) for num in range(2**n)]).astype(int)
    nt_x, applied = nt.evaluate_sigmas(nt_s.sigmas, all_data)
    me_s = me_policy(hmm)
    nt_res = exact_rates(hmm, n, nt_s, verbosity=2)
    me_res = exact_rates(hmm, n, me_s, verbosity=2)
    assert nt_res[b"permuted_log_rate"] < me_res[b"permuted_log_rate"]
    nontriv = (nt_res[b'permuted_extras'][b'perms'] != torch.arange(2)).any(-1)
    total_applied = (nontriv.int().sum(-1) % 2).bool()
    x = nt_res[b'permuted_extras'][b'history']['x'][:, -1, :].clone().detach()
    assert np.all(nontriv.numpy()[..., :-1] == applied[..., 1:])
    x[total_applied] = -x[total_applied]
    nt_plisd = nx_to_log_probs(x, m_to_r(pq_to_m(p, q))).transpose()
    res_plisd = nt_res[b'permuted_extras'][b'posterior_log_initial_state_dist'].numpy()
    assert np.allclose(np.exp(nt_plisd), np.exp(res_plisd))
