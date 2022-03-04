import pytest
import unittest
from copy import deepcopy
import numpy as np
import torch
import pyro.distributions as dist
from perm_hmm.models.hmms import DiscreteHMM, PermutedDiscreteHMM
from perm_hmm.policies.min_tree import MinEntPolicy
from perm_hmm.util import bin_ent, ZERO, perm_idxs_from_perms


def get_marginals_and_conditionals(perm_policy):
    state = perm_policy.belief_state.transition(perm_policy.possible_perms)
    cur_cond_init = state.logits - state.logits.logsumexp(-1, keepdim=True)
    prior_log_current = state.logits.logsumexp(-2)
    prior_log_inits = state.logits.logsumexp(-1)[:, 0, :]
    return prior_log_inits, prior_log_current, cur_cond_init


def setUp(self):
    self.observation_probs = torch.tensor([.5, 1])
    self.observation_dist = dist.Bernoulli(self.observation_probs)
    self.possible_perms = torch.tensor([[0, 1],
                                        [1, 0]], dtype=int)
    self.transition_logits = torch.tensor([[1-ZERO, ZERO], [.5, .5]]).log().float()
    self.initial_logits = torch.tensor([.5, .5]).log()
    self.bdhmm = PermutedDiscreteHMM(self.initial_logits,
                                     self.transition_logits,
                                     self.observation_dist)
    self.shmm = DiscreteHMM(self.initial_logits,
                            self.transition_logits,
                            self.observation_dist)
    self.perm_policy = MinEntPolicy(self.possible_perms, self.bdhmm)
    self.integration_time = 1


transition_logits = torch.tensor([[1 - ZERO, ZERO], [.5, .5]]).log()
initial_logits = torch.tensor([.5, .5]).log()
observation_probs = torch.tensor([.5, 1])
observation_dist = dist.Bernoulli(observation_probs)
my_hmm = PermutedDiscreteHMM(initial_logits, transition_logits, observation_dist)
pyro_hmm = DiscreteHMM(initial_logits, transition_logits, observation_dist)
possible_perms = torch.tensor([[0, 1], [1, 0]], dtype=int)

data_0 = torch.tensor([1.0, 1, 0])
data_1 = torch.tensor([1, 1.0, 0, 1, 0, 0])
data_2 = torch.tensor([1.0, 1, 1])
data_3 = torch.tensor([0.0, 1, 1])



@pytest.mark.parametrize('data,hmm', [
    (data_0, my_hmm),
    (data_1, my_hmm),
    (data_0, pyro_hmm),
    (data_2, my_hmm),
    (data_3, my_hmm),
])
def test_posterior_init(data, hmm):
    perm_policy = MinEntPolicy(possible_perms, my_hmm)
    perm_policy.reset(save_history=True)
    idperms = possible_perms[torch.zeros(data.shape[0], dtype=int)]
    penultimates = perm_policy.penultimates_from_sequence(data, idperms)
    plisd = penultimates[-1].logsumexp(-1)
    spinit = hmm.posterior_log_initial_state_dist(data)
    assert plisd.allclose(spinit)


@pytest.mark.parametrize('hmm', [
    my_hmm,
])
def test_data_stack(hmm):
    all_data = torch.tensor([list(map(int, ("{{0:0{}b}}".format(6)).format(j))) for j in range(2 ** 6)], dtype=torch.float)
    slp = hmm.log_prob(all_data)
    sllp = torch.tensor([hmm.log_prob(data) for data in all_data])
    assert slp.allclose(sllp)


def test_no_perms():
    data = data_0
    perm_policy = MinEntPolicy(possible_perms, my_hmm)
    perm_policy.reset(save_history=True)
    data_idx = perm_policy.data_to_idx(data)

    trivial_perm = torch.arange(perm_policy.hmm.initial_logits.shape[0])
    trivial_perm_idx = perm_idxs_from_perms(possible_perms, trivial_perm)

    hand_calc_inits = [
        torch.tensor([1 / 3, 2 / 3]),
        torch.tensor([1 / 4, 3 / 4]),
        torch.tensor([1 / 3, 2 / 3]),
    ]
    hand_calc_currents = [
        torch.tensor([2 / 3, 1 / 3]),
        torch.tensor([3 / 4, 1 / 4]),
        torch.tensor([1, ZERO]),
    ]
    hand_calc_current_cond_inits = [
        transition_logits.exp(),
        torch.tensor([[1, ZERO], [2 / 3, 1 / 3]]),
        torch.tensor([[1, ZERO], [1, ZERO]]),
    ]
    for step, i, c, cci in zip(range(3), hand_calc_inits, hand_calc_currents, hand_calc_current_cond_inits):
        perm_policy.tree.prune_tree(data_idx[step])
        perm_policy.tree.grow()

        state = perm_policy.tree.beliefs[-2]
        current = state.logits.logsumexp(-2)[trivial_perm_idx, 0, :]
        assert current.exp().allclose(c, atol=1e-7)
        init = state.logits.logsumexp(-1)[trivial_perm_idx, 0, :]
        assert init.exp().allclose(i, atol=1e-7)
        cur_cond_init = state.logits - state.logits.logsumexp(-1, keepdim=True)
        cur_cond_init = cur_cond_init[trivial_perm_idx, 0, :, :]
        assert cur_cond_init.exp().allclose(cci, atol=1e-7)

        perm_policy.tree.prune_tree(trivial_perm_idx)


def test_with_perm():
    perms = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=int)
    perm_idxs = perm_idxs_from_perms(possible_perms, perms)
    perm_policy = MinEntPolicy(possible_perms, my_hmm, save_history=True)
    data = data_0
    data_idxs = perm_policy.data_to_idx(data)

    hand_calc_inits = [
        torch.tensor([1 / 3, 2 / 3]),
        torch.tensor([3 / 7, 4 / 7]),
        torch.tensor([1 / 3, 2 / 3]),
    ]
    hand_calc_currents = [
        torch.tensor([5 / 6, 1 / 6]),
        torch.tensor([6 / 7, 1 / 7]),
        torch.tensor([1 - ZERO, ZERO]),
    ]
    hand_calc_cur_cond_inits = [
        transition_logits[perms[0]].exp(),
        torch.tensor([[2 / 3, 1 / 3], [1 - ZERO, ZERO]]),
        torch.tensor([[1 - ZERO, ZERO], [1 - ZERO, ZERO]]),
    ]

    for step, i, c, cci in zip(range(3), hand_calc_inits, hand_calc_currents, hand_calc_cur_cond_inits):
        data_idx = data_idxs[step]
        perm_policy.tree.prune_tree(data_idx)
        perm_policy.tree.grow()

        state = perm_policy.tree.beliefs[-2]
        perm_idx = perm_idxs[step]
        current = state.logits.logsumexp(-2)[perm_idx, 0, :]
        assert current.exp().allclose(c, atol=1e-7)
        init = state.logits.logsumexp(-1)[perm_idx, 0, :]
        assert init.exp().allclose(i, atol=1e-7)
        cur_cond_init = state.logits - state.logits.logsumexp(-1, keepdim=True)
        cur_cond_init = cur_cond_init[perm_idx, 0, :, :]
        assert cur_cond_init.exp().allclose(cci, atol=1e-7)

        perm_policy.tree.prune_tree(perm_idx)


def test_with_all_vals():
    perms = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=int)
    perm_idxs = perm_idxs_from_perms(possible_perms, perms)
    perm_policy = MinEntPolicy(possible_perms, my_hmm, save_history=True)
    data = data_0
    data_idxs = perm_policy.data_to_idx(data)

    perm_policy.tree.prune_tree(data_idxs[0])
    perm_policy.tree.grow()

    plid = perm_policy.tree.beliefs[-1].logits.logsumexp(-1)[perm_idxs[0], :, 0, :]

    t1 = plid.exp()
    t2 = torch.tensor([[1/2, 1/2], [1/4, 3/4]])
    assert t1.allclose(t2, atol=3e-07)
    perm_policy.tree.prune_tree(perm_idxs[0])

    perm_policy.tree.prune_tree(data_idxs[1])
    perm_policy.tree.grow()
    state = perm_policy.tree.beliefs[-2].logits[perm_idxs[1], 0, :, :]
    cur_cond_init = state - state.logsumexp(-1, keepdim=True)
    prior_log_current = state.logsumexp(-2)
    assert prior_log_current.exp().allclose(torch.tensor(
        [3/4, 1/4]
    ), atol=3e-07)
    assert cur_cond_init.exp().allclose(torch.tensor(
        [[1/2, 1/2],
         [5/6, 1/6]]
    ), atol=3e-07)


def test_entropies():
    # perms = torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1]], dtype=int)
    # perm_idxs = perm_idxs_from_perms(possible_perms, perms)
    perm_policy = MinEntPolicy(possible_perms, my_hmm, save_history=True)
    data = data_0
    data_idxs = perm_policy.data_to_idx(data)
    perm_policy.reset(save_history=False)
    # ent = perm_policy.expected_entropy()
    # should_ent = 3/4*bin_ent(torch.tensor([2/3]).log())
    # assertTrue(ent[0].allclose(should_ent))
    perm_policy.tree.prune_tree(data_idxs[0])
    perm_policy.tree.grow()
    perm_choice, ent = perm_policy.tree.perm_idxs_from_log_cost(perm_policy.log_cost_func, return_log_costs=True)
    should_ent = 5/12*bin_ent(torch.tensor([1/5]).log()) + 7/12*bin_ent(torch.tensor([3/7]).log())
    assert ent[-2][1, 0].exp().allclose(should_ent)
    should_ent = 1/3*bin_ent(torch.tensor([1/2]).log()) + 2/3*bin_ent(torch.tensor([1/4]).log())
    assert ent[-2][0, 0].exp().allclose(should_ent)


def test_reset():
    data = data_0
    perm_policy = MinEntPolicy(possible_perms, my_hmm, save_history=True)
    copy_policy = deepcopy(perm_policy)
    num_states = initial_logits.shape[0]
    idperm = torch.arange(0, num_states, dtype=int)
    for datum in data:
        perm_policy.tree.prune_tree(perm_policy.data_to_idx(datum))
        perm_policy.tree.grow()
        perm_policy.tree.prune_tree(perm_idxs_from_perms(possible_perms, idperm))
    perm_policy.reset()
    assert all([torch.all(belief.logits.eq(copy_belief.logits)) for belief, copy_belief in zip(perm_policy.tree.beliefs, copy_policy.tree.beliefs)])
