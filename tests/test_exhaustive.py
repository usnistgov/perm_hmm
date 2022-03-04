import pytest

from operator import mul
from functools import reduce
import numpy as np
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import torch
import pyro.distributions as dist

import adapt_hypo_test.two_states.util as twotil

from perm_hmm.models.hmms import PermutedDiscreteHMM, random_phmm
from perm_hmm.util import id_and_transpositions, all_strings, log1mexp
from perm_hmm.policies.ignore_transitions import IgnoreTransitions
from perm_hmm.policies.exhaustive import ExhaustivePolicy
from perm_hmm.simulator import HMMSimulator
from perm_hmm.policies.belief_tree import HMMBeliefTree

ZERO = 1e-100


@pytest.mark.parametrize("p,q,steps", [(0.1, 0.3, 8), (0.1, 0.2, 8), (0.01, 0.8, 8)])
def test_exhaustive_value(p, q, steps):
    print("p = {}, q = {}".format(p, q))
    il = (torch.ones(2)/2).log()
    observation_dist = dist.Bernoulli(torch.tensor([p, 1-q]))
    transition_logits = torch.tensor([[1, ZERO], [ZERO, 1]]).log()
    hmm = PermutedDiscreteHMM(il, transition_logits, observation_dist)
    possible_perms = torch.tensor([[0, 1], [1, 0]])
    es = ExhaustivePolicy(possible_perms, hmm, steps)
    costs = es.compute_perm_tree(return_log_costs=True, delete_belief_tree=False)
    vvv = log1mexp(costs[0].ravel()).exp().numpy().item()
    sim = HMMSimulator(hmm)
    ep, ed = sim.all_classifications(steps, perm_policy=es, verbosity=1)
    eperms = ed[b'perms'].numpy()
    ve = ep.log_misclassification_rate().exp().numpy()
    it = IgnoreTransitions(possible_perms, p, q, 0, 1)
    v = it.solve(steps)
    ip, idict = sim.all_classifications(steps, perm_policy=it, verbosity=2)
    r = twotil.m_to_r(twotil.pq_to_m(p, q))
    istories = twotil.nx_to_log_odds(idict[b'history']['x'].numpy(), r)
    plt.plot(istories.transpose())
    plt.show()
    beliefs = []
    for i in range(steps):
        lps = es.belief_tree.beliefs[-1-2*i].logits
        perms = ed[b'perms'][torch.arange(2**(steps-i))*2**i]
        perms = perms[:, :steps-i]
        b = select_perm_path(lps, es.possible_perms, perms)
        for j in range(i):
            b = b.unsqueeze(-3)
            b = torch.tile(b, (1,)*(len(b.shape) - 3) + (2, 1, 1))
        b = b.reshape((-1, 2, 2))
        beliefs.append(b)
    beliefs = beliefs[::-1]
    odds = []
    for b in beliefs:
        b = b.logsumexp(-1).numpy()
        odds.append(b[..., 1] - b[..., 0])
    odds = np.stack(odds)
    plt.plot(odds)
    plt.show()
    possible_lps = twotil.lp_grid(steps, twotil.m_to_r(twotil.pq_to_m(p, q)))
    assert np.all(np.any(np.all(np.isclose(np.exp(possible_lps.reshape(2, -1)[:, None, :]), np.exp(es.belief_tree.beliefs[-1].logits.logsumexp(-1).squeeze(-2).reshape((-1, 2)).transpose(0, 1).numpy()[:, :, None]), atol=1e-7), axis=0), axis=-1))
    iperms = idict[b'perms'].numpy()
    # assert np.all(eperms == iperms)
    v = np.exp(twotil.log1mexp((logsumexp(v.ravel()) - np.log(2)).item()))
    print ("vvv = {}".format(vvv))
    print("ve = {}".format(ve))
    print("v = {}".format(v))
    b = select_perm_path(es.belief_tree.beliefs[-1].logits, es.possible_perms, ed[b'perms'])
    s = all_strings(steps)
    log_min_entropy = (-(b.logsumexp(-1).max(-1)[0])).log()
    lps = hmm.log_prob(s, ed[b'perms'])
    rrrate = log1mexp((lps + (- (log_min_entropy.exp()))).logsumexp(-1)).exp().numpy()
    assert np.allclose(rrrate, v, atol=1e-7)
    assert np.allclose(ve, v, atol=1e-7)
    assert np.allclose(vvv, v, atol=5e-7)


def indices_of(perm, possible_perms):
    retval = (perm.unsqueeze(-2) == possible_perms).all(-1)
    assert (retval.sum(-1) == 1).all()
    return retval.long().argmax(-1)


def select_perm_path(lps, possible_perms, perms):
    height = len(lps.shape) - 3
    retval = torch.moveaxis(
        lps,
        tuple(range(height)),
        tuple((i // 2) if i % 2 == 0 else (i // 2 + height // 2 + 1) for i in
              range(height))
    )
    retval = torch.reshape(retval, (
        reduce(mul, retval.shape[:height // 2+1], 1),) + retval.shape[height // 2+1:])
    for i in range(perms.shape[-2]-1):
        perm = perms[..., i, :]
        pidx = indices_of(perm, possible_perms)
        retval = retval[torch.arange(retval.shape[0]), pidx]
    return retval.squeeze(-3)


@pytest.mark.parametrize("possible_perms,hmm,steps", [
    (id_and_transpositions(3), random_phmm(3), 4),
])
def test_exhaustive_value_brute(possible_perms, hmm, steps):
    es = ExhaustivePolicy(possible_perms, hmm, steps)
    es.compute_perm_tree(delete_belief_tree=False)
    sim = HMMSimulator(hmm)
    ep, d = sim.all_classifications(steps, perm_policy=es, verbosity=1)
    final_belief = select_perm_path(es.belief_tree.beliefs[-1].logits, es.possible_perms, d[b'perms'])
    assert torch.allclose(final_belief.logsumexp(-1).exp(), d[b'posterior_log_initial_state_dist'].exp(), atol=1e-7)
    assert torch.all(final_belief.logsumexp(-1).argmax(-1) == ep.classifications)
    brute_force_rate = ep.log_misclassification_rate()
    s = all_strings(steps)
    assert torch.allclose(ep.log_joint.logsumexp(-2), hmm.log_prob(s, d[b'perms']))
    after_rate = log1mexp((ep.log_joint.logsumexp(-2) + final_belief.logsumexp(-1).max(-1)[0]).logsumexp(-1))
    log_min_entropy = (-(final_belief.logsumexp(-1).max(-1)[0])).log()
    rrrate = log1mexp((hmm.log_prob(s, d[b'perms']) + d[b'posterior_log_initial_state_dist'].max(-1)[0]).logsumexp(-1))
    lps = hmm.log_prob(s, d[b'perms'])
    rrate = log1mexp((lps + (- (log_min_entropy.exp()))).logsumexp(-1))
    assert torch.allclose(after_rate.exp(), rrrate.exp())
    assert torch.allclose(brute_force_rate.exp(), rrrate.exp())
    assert torch.allclose(rrate.exp(), rrrate.exp())
