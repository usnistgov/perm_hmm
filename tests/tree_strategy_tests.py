import pytest
import numpy as np
import torch
import pyro.distributions as dist
from example_systems.three_states import three_state_hmm
from perm_hmm.models.hmms import PermutedDiscreteHMM
from perm_hmm.util import all_strings, id_and_transpositions, ZERO
from tests.min_ent import MinEntropyPolicy
from perm_hmm.policies.min_tree import MinTreePolicy
import perm_hmm.log_cost as cf


def simple_hmm():
    observation_probs = torch.tensor([.5, 1])
    observation_dist = dist.Bernoulli(observation_probs)
    possible_perms = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=int)
    transition_logits = torch.tensor([[1 - ZERO, ZERO], [.5, .5]]).log()
    initial_logits = torch.tensor([.5, .5]).log()
    hmm = PermutedDiscreteHMM(initial_logits, transition_logits, observation_dist)
    return hmm, possible_perms


@pytest.mark.parametrize("hmm,possible_perms,num_steps",[
    simple_hmm() + (4,),
    (three_state_hmm(-3, -4), id_and_transpositions(3), 4),
    (three_state_hmm(-1, -4), id_and_transpositions(3), 4),
    (three_state_hmm(-5, -4), id_and_transpositions(3), 4),
    ])
def test_posterior_distributions(hmm, possible_perms, num_steps):
    num_states = hmm.initial_logits.shape[0]
    all_data = all_strings(num_steps, num_states)
    mes2 = MinEntropyPolicy(possible_perms, hmm, save_history=True)
    mes1 = MinTreePolicy(possible_perms, hmm, cf.log_initial_entropy, 1, initialize_tree=False)
    mes1.initialize_tree(1, data_len=all_data.shape[0])
    reverse_perm_dict = {tuple(v): k for k, v in enumerate(possible_perms.numpy().tolist())}
    for j in range(num_steps):
        mes1.tree.prune_tree(mes1.data_to_idx(all_data[..., j]))
        mes1.tree.grow(mes1.possible_perms)
        b = mes1.tree[-1]

        mes2.belief_state = mes2.belief_state.bayes_update(all_data[..., j])
        assert np.allclose(mes2.belief_state.logits.exp().double().numpy(), mes1.tree.beliefs[0].logits.exp().double().numpy(), atol=1e-6)
        pl2 = mes2.distributions_for_all_perms()
        after_transition_2 = pl2.logsumexp(-1)
        after_transition_1 = mes1.tree[1].logits.transpose(0, 1)
        assert np.allclose(after_transition_1.exp().double().numpy(), after_transition_2.exp().double().numpy(), atol=1e-6)
        pl2 -= pl2.logsumexp(-3, keepdim=True).logsumexp(-2, keepdim=True)
        pl2 = torch.from_numpy(np.moveaxis(pl2.numpy(), (-1, -2, -3, -4, -5), (-4,-1,-2,-5,-3)))
        assert torch.allclose(pl2.exp().double(), b.logits.exp().double(), atol=1e-6)

        perm = mes2.calculate_perm_from_belief(return_dict=False)
        perm_idx = torch.tensor([reverse_perm_dict[tuple(p.numpy())] for p in perm])
        mes1.tree.prune_tree(perm_idx)
        mes2.belief_state = mes2.belief_state.transition(perm.unsqueeze(-2))


@pytest.mark.parametrize("hmm,possible_perms,num_steps",[
    simple_hmm() + (4,),
    (three_state_hmm(-3, -4), id_and_transpositions(3), 4),
    (three_state_hmm(-1, -4), id_and_transpositions(3), 4),
    (three_state_hmm(-5, -4), id_and_transpositions(3), 4),
    ])
def test_posterior_entropy(hmm, possible_perms, num_steps):
    num_states = hmm.initial_logits.shape[0]
    all_data = all_strings(num_steps, num_states)
    mes2 = MinEntropyPolicy(possible_perms, hmm, save_history=True)
    mes1 = MinTreePolicy(possible_perms, hmm, cf.log_initial_entropy, 1, initialize_tree=False, save_history=True)
    mes1.initialize_tree(1, data_len=all_data.shape[0])
    reverse_perm_dict = {tuple(v): k for k, v in enumerate(possible_perms.numpy().tolist())}
    for j in range(num_steps):
        mes2.belief_state = mes2.belief_state.bayes_update(all_data[..., j])
        mes1.tree.prune_tree(mes1.data_to_idx(all_data[..., j]))
        mes1.tree.grow(mes1.possible_perms)
        perm_tree, costs = mes1.tree.perm_idxs_from_log_cost(mes1.log_cost_func, return_costs=True)
        entropy2, distn2 = mes2.cond_entropies_for_all_perms(return_distn=True)
        distn1 = hmm.observation_dist.log_prob(hmm.enumerate_support(expand=False))
        yk = (mes1.tree[-2].logits.logsumexp(-2).unsqueeze(-3) + distn1.unsqueeze(-2)).logsumexp(-1)
        distn1 = (yk.unsqueeze(-1).unsqueeze(-2) + mes1.tree[-1].logits)
        distn2 = torch.tensor(np.moveaxis(distn2.numpy(), (-1, -2, -3, -4, -5), (-4,-1,-2,-5,-3)))
        assert torch.allclose(distn1.exp().double(), distn2.exp().double(), atol=1e-6)
        assert torch.allclose(costs[-2], (costs[-1] + yk).logsumexp(-2), atol=1e-6)
        s1yk = distn1.logsumexp(-1)
        jointent1 = -(s1yk.exp()*s1yk).sum(-3).sum(-1)
        s1 = s1yk.logsumexp(-1)
        yent1 = -(s1.exp()*s1).sum(-2)
        condent1 = (jointent1 - yent1).transpose(0, 1)
        assert torch.allclose(entropy2.double(), condent1.double(), atol=1e-6)
        plisd1 = mes1.tree[-1].logits.logsumexp(-1)
        log_postinitent = (-(plisd1.exp()*plisd1).sum(-1)).log()
        post_ent1 = (yk + log_postinitent).logsumexp(-2).exp().transpose(0, 1)
        assert torch.allclose(post_ent1.double(), entropy2.double(), atol=1e-6)
        entropy1 = costs[-2].transpose(0, 1).exp()
        assert torch.allclose(entropy1.double(), entropy2.double(), atol=1e-6)
        perm = mes2.calculate_perm_from_belief(return_dict=False)
        perm_idx = torch.tensor([reverse_perm_dict[tuple(p.numpy())] for p in perm])
        mes1.tree.prune_tree(perm_idx)
        mes2.belief_state = mes2.belief_state.transition(perm.unsqueeze(-2))


# @pytest.mark.parametrize("n_states", [2, 3, 4])
# def test_min_tree_min_ent(n_states):
#     n_steps = 5
#     hmm = random_phmm(n_states)
#     allowed_permutations = id_and_transpositions(n_states)
#     mes1 = MinEntropyPolicy(allowed_permutations, hmm)
#     mes2 = MinTreePolicy(allowed_permutations, hmm, vf.negative_min_entropy, 1)
#     all_data = all_strings(n_steps)
#     perms1 = mes1.get_perms(all_data)
#     perms2 = mes2.get_perms(all_data)
#     assert (perms1 == perms2).all()
