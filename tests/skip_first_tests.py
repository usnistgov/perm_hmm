import numpy as np
import torch
import pyro.distributions as dist
from perm_hmm.models.hmms import SkipFirstDiscreteHMM
from perm_hmm.util import num_to_data, all_strings


def state_sequence_lp(seq, il, tl):
    n = len(seq) - 1
    return il[seq[0]] + tl.expand((n,) + tl.shape)[
        torch.arange(n), seq[:-1], seq[1:]].sum(-1)


def log_joint_at_seq(data, il, tl, od, seq):
    n = len(data)
    retval = state_sequence_lp(seq, il, tl)
    retval += od.log_prob(data[:, None])[torch.arange(n), seq[1:]].sum(-1)
    return retval


def brute_force_skip_first_lp(data, il, tl, od):
    n = len(data)
    retval = -float('inf')
    nstates = len(il)
    for seq in all_strings(n+1, base=nstates, dtype=int):
        retval = np.logaddexp(retval, log_joint_at_seq(data, il, tl, od, seq).numpy())
    return retval


def brute_force_skip_first_jog_joint(data, il, tl, od, i):
    n = len(data)
    retval = -float('inf')
    nstates = len(il)
    for seq in all_strings(n, base=nstates, dtype=int):
        seq = torch.cat((torch.tensor([i]), seq))
        retval = np.logaddexp(retval, log_joint_at_seq(data, il, tl, od, seq).numpy())
    return retval


def test_skipfirst_logprob():
    n = 3
    tmax = 5
    dirichlet = dist.Dirichlet(torch.ones(n) / n)
    initial_logits = (torch.ones(n) / n).log()
    transition_logits = dirichlet.sample((n,)).log()
    observation_dist = dist.Bernoulli(torch.rand(n))
    sfhmm = SkipFirstDiscreteHMM(initial_logits, transition_logits, observation_dist)
    i = torch.randint(2**tmax, (1,))
    data = num_to_data(i, tmax)
    sflp = sfhmm.log_prob(data)
    bfsflp = brute_force_skip_first_lp(data, initial_logits, transition_logits, observation_dist)
    assert sflp.isclose(torch.tensor(bfsflp))


def test_total_skipfirst_logprob():
    n = 3
    tmax = 5
    dirichlet = dist.Dirichlet(torch.ones(n) / n)
    initial_logits = (torch.ones(n) / n).log()
    transition_logits = dirichlet.sample((n,)).log()
    observation_dist = dist.Bernoulli(torch.rand(n))
    sfhmm = SkipFirstDiscreteHMM(initial_logits, transition_logits, observation_dist)
    data = all_strings(5)
    assert torch.isclose(sfhmm.log_prob(data).logsumexp(-1), torch.tensor(0.).double(), atol=5e-7)


def test_skipfirst_plisd():
    n = 3
    tmax = 5
    dirichlet = dist.Dirichlet(torch.ones(n) / n)
    initial_logits = (torch.ones(n) / n).log()
    transition_logits = dirichlet.sample((n,)).log()
    observation_dist = dist.Bernoulli(torch.rand(n))
    sfhmm = SkipFirstDiscreteHMM(initial_logits, transition_logits, observation_dist)
    i = torch.randint(2**tmax, (1,))
    data = num_to_data(i, tmax)
    sfplisd = sfhmm.posterior_log_initial_state_dist(data)
    bfsfplisd = torch.tensor([brute_force_skip_first_jog_joint(data, initial_logits, transition_logits, observation_dist, i) for i in range(n)]) - sfhmm.log_prob(data)
    assert sfplisd.allclose(bfsfplisd.float())


def test_skipfirst_plisd_all():
    n = 3
    tmax = 5
    dirichlet = dist.Dirichlet(torch.ones(n) / n)
    initial_logits = (torch.ones(n) / n).log()
    transition_logits = dirichlet.sample((n,)).log()
    observation_dist = dist.Bernoulli(torch.rand(n))
    sfhmm = SkipFirstDiscreteHMM(initial_logits, transition_logits, observation_dist)
    i = torch.randint(2**tmax, (1,))
    data = all_strings(tmax)
    sfplisd = sfhmm.posterior_log_initial_state_dist(data)
    bfsfplisd = torch.tensor([
        [brute_force_skip_first_jog_joint(dat, initial_logits, transition_logits, observation_dist, i) for i in range(n)]
        for dat in data
    ])
    bfsfplisd = bfsfplisd - sfhmm.log_prob(data)[:, None]
    assert sfplisd.allclose(bfsfplisd.float())

