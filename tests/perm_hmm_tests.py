import numpy as np

import torch
import pyro.distributions as dist

from perm_hmm.models.hmms import PermutedDiscreteHMM
from perm_hmm.policies.policy import PermPolicy
from perm_hmm.policies.min_tree import MinEntPolicy
from perm_hmm.policies.rotator_policy import RotatorPolicy, cycles
from perm_hmm.util import ZERO, all_strings, num_to_data, id_and_transpositions


class Shifter(PermPolicy):

    def __init__(self, hmm, save_history=False):
        self.num_states = hmm.initial_logits.shape[0]
        possible_perms = cycles(self.num_states)
        super().__init__(possible_perms, save_history=save_history)

    def reset(self, save_history=False):
        super().reset(save_history=save_history)

    def calculate_perm(self, data: torch.Tensor) -> (torch.Tensor, dict):
        return self.possible_perms[data.long()], {}


def test_trivial_sample():
    """
    Check that the PermutedDiscreteHMM returns the expected sample for a
    permutation by checking a trivial model.
    """
    initial_logits = torch.tensor([1.-ZERO, ZERO, ZERO, ZERO, ZERO]).log()
    initial_logits -= initial_logits.logsumexp(-1)
    transition_logits = torch.full((5, 5), ZERO)
    transition_logits += torch.eye(5)
    transition_logits = transition_logits.log()
    transition_logits -= transition_logits.logsumexp(-1)
    output_probs = transition_logits.clone().detach().exp()
    observation_dist = dist.Categorical(output_probs)
    hmm = PermutedDiscreteHMM(initial_logits, transition_logits, observation_dist)
    perm_policy = RotatorPolicy(hmm)
    perm_policy.reset()
    s, o = hmm.sample((5,), perm_policy)
    assert s.allclose(torch.arange(5))
    assert o.allclose(torch.arange(5))


def test_less_trivial_sample():
    initial_logits = torch.tensor([1.-ZERO, ZERO, ZERO, ZERO, ZERO]).log()
    initial_logits -= initial_logits.logsumexp(-1)
    observation_logits = torch.full((5, 5), ZERO)
    observation_logits += torch.eye(5)
    observation_logits = observation_logits.log()
    observation_logits -= observation_logits.logsumexp(-1)
    output_probs = observation_logits.exp()
    observation_dist = dist.Categorical(output_probs)
    transition_logits = observation_logits.clone().detach()
    transition_logits = torch.roll(transition_logits, -1, dims=0)
    hmm = PermutedDiscreteHMM(initial_logits, transition_logits, observation_dist)
    perm_policy = RotatorPolicy(hmm)
    s, o = hmm.sample((5,), perm_policy)
    assert s.allclose((2*torch.arange(5)) % 5)
    assert o.allclose((2*torch.arange(5)) % 5)


def test_data_dependent():
    initial_logits = torch.tensor([1.-ZERO, ZERO, ZERO, ZERO, ZERO]).log()
    initial_logits -= initial_logits.logsumexp(-1)
    observation_logits = torch.full((5, 5), ZERO)
    observation_logits += torch.eye(5)
    observation_logits = observation_logits.log()
    observation_logits -= observation_logits.logsumexp(-1)
    output_probs = observation_logits.exp()
    observation_dist = dist.Categorical(output_probs)
    transition_logits = observation_logits.clone().detach()
    transition_logits = torch.roll(transition_logits, -1, dims=0)
    hmm = PermutedDiscreteHMM(initial_logits, transition_logits, observation_dist)
    perm_policy = Shifter(hmm)
    s, o = hmm.sample((5,), perm_policy)
    op = torch.empty_like(o)
    op[0] = 0
    for i in range(1, 5):
        op[i] = (2*op[i-1]+1) % 5
    assert o.allclose(op)
    perm_policy.reset()
    s, o = hmm.sample((10, 5), perm_policy)
    assert o.allclose(op)


def test_posterior_log_initial_state_dist():
    observation_probs = torch.tensor([.5, 1])
    observation_dist = dist.Bernoulli(observation_probs)
    possible_perms = torch.tensor([[0, 1], [1, 0]], dtype=int)
    transition_logits = torch.tensor([[1 - ZERO, ZERO], [.5, .5]]).log().float()
    initial_logits = torch.tensor([.5, .5]).log()
    hmm = PermutedDiscreteHMM(initial_logits, transition_logits, observation_dist)
    perm_policy = MinEntPolicy(possible_perms, hmm)
    # data = torch.tensor([1, 1.0, 0, 1, 0, 0])
    data = torch.tensor([
        [1.0, 1, 1, 1],
        [1.0, 0, 0, 0],
        [1.0, 1, 0, 0],
    ])
    # data = torch.tensor([0.0, 1, 1])
    perm_policy.reset(save_history=True)
    perms = perm_policy.get_perms(data)
    d = perm_policy.calc_history
    da = perm_policy.tree[0].logits.logsumexp(-1)
    dap = hmm.posterior_log_initial_state_dist(data, perms)
    assert torch.allclose(dap, da)


def apply_perm(seq, perm):
    return perm[torch.arange(len(seq)), seq]


def state_sequence_lp(seq, il, tl, perm):
    n = len(seq) - 1
    perm_seq = apply_perm(seq, perm)
    return il[seq[0]] + tl.expand((n,) + tl.shape)[
        torch.arange(n), perm_seq[:-1], seq[1:]
    ].sum(-1)


def log_joint_at_seq(data, il, tl, od, seq, perm):
    n = len(data)
    retval = state_sequence_lp(seq, il, tl, perm)
    retval += od.log_prob(data[:, None])[torch.arange(n), seq].sum(-1)
    return retval


def brute_force_lp(data, il, tl, od, perm):
    n = len(data)
    retval = -float('inf')
    nstates = len(il)
    for seq in all_strings(n, base=nstates, dtype=int):
        retval = np.logaddexp(retval, log_joint_at_seq(data, il, tl, od, seq, perm).numpy())
    return retval


def brute_force_jog_joint(data, il, tl, od, i, perm):
    n = len(data)
    retval = -float('inf')
    nstates = len(il)
    for seq in all_strings(n, base=nstates, dtype=int):
        seq = torch.cat((torch.tensor([i]), seq))
        retval = np.logaddexp(retval, log_joint_at_seq(data, il, tl, od, seq, perm).numpy())
    return retval


def test_perm_log_prob():
    n = 3
    tmax = 5
    dirichlet = dist.Dirichlet(torch.ones(n) / n)
    initial_logits = (torch.ones(n) / n).log()
    transition_logits = dirichlet.sample((n,)).log()
    observation_dist = dist.Bernoulli(torch.rand(n))
    hmm = PermutedDiscreteHMM(initial_logits, transition_logits, observation_dist)
    perm_policy = MinEntPolicy(id_and_transpositions(n), hmm)
    i = torch.randint(2**tmax, (1,))
    data = num_to_data(i, tmax)
    data = data.unsqueeze(-2)
    perms = perm_policy.get_perms(data)
    lp = hmm.log_prob(data, perms)
    bflp = brute_force_lp(data.squeeze(), initial_logits, transition_logits, observation_dist, perms.squeeze())
    assert lp.double().isclose(torch.tensor(bflp))


def test_total_logprob():
    n = 3
    tmax = 3
    dirichlet = dist.Dirichlet(torch.ones(n) / n)
    initial_logits = (torch.ones(n) / n).log()
    transition_logits = dirichlet.sample((n,)).log()
    observation_dist = dist.Bernoulli(torch.rand(n))
    hmm = PermutedDiscreteHMM(initial_logits, transition_logits, observation_dist)
    perm_policy = MinEntPolicy(id_and_transpositions(n), hmm)
    data = all_strings(tmax)
    perms = perm_policy.get_perms(data)
    lp = hmm.log_prob(data, perms)
    assert torch.isclose(lp.double().logsumexp(-1), torch.tensor(0.).double(), atol=5e-7)
    bflp = torch.zeros_like(lp)
    for i, (dat, perm) in enumerate(zip(data, perms)):
        bflp[i] = brute_force_lp(dat.squeeze(), initial_logits, transition_logits, observation_dist, perm.squeeze())
    assert lp.double().allclose(bflp.double(), atol=5e-7)


