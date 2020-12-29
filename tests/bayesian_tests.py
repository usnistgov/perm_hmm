import unittest
from copy import deepcopy
import torch
import pyro.distributions as dist
from perm_hmm.models.hmms import DiscreteHMM, PermutedDiscreteHMM
from perm_hmm.strategies.min_ent import MinEntropySelector
from perm_hmm.util import bin_ent, ZERO


class BdhmmTestCase(unittest.TestCase):
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
        self.perm_selector = MinEntropySelector(self.possible_perms, self.bdhmm)
        self.data = torch.tensor([1.0, 1, 0])
        self.data_1 = torch.tensor([1, 1.0, 0, 1, 0, 0])
        self.data_2 = torch.tensor([1.0, 1, 1])
        self.data_3 = torch.tensor([0.0, 1, 1])
        self.integration_time = 1

    def test_posterior_init(self):
        self.perm_selector.reset(save_history=False)
        idperms = self.possible_perms[torch.zeros(self.data.shape[0], dtype=int)]
        dh = self.perm_selector._calculate_dists(self.data, idperms)
        spinit = self.shmm.posterior_log_initial_state_dist(self.data)
        self.assertTrue(dh[-1].allclose(spinit))
        idperms = self.possible_perms[torch.zeros(self.data_2.shape[0], dtype=int)]
        dh = self.perm_selector._calculate_dists(self.data_2, idperms)
        spinit = self.shmm.posterior_log_initial_state_dist(self.data_2)
        self.assertTrue(dh[-1].allclose(spinit))
        idperms = self.possible_perms[torch.zeros(self.data_3.shape[0], dtype=int)]
        dh = self.perm_selector._calculate_dists(self.data_3, idperms)
        spinit = self.shmm.posterior_log_initial_state_dist(self.data_3)
        self.assertTrue(dh[-1].allclose(spinit))
        all_data = torch.tensor([list(map(int, ("{{0:0{}b}}".format(6)).format(j))) for j in range(2 ** 6)], dtype=torch.float)
        slp = self.shmm.log_prob(all_data)
        sllp = torch.tensor([self.shmm.log_prob(data) for data in all_data])
        self.assertTrue(slp.allclose(sllp))

    def test_no_perms(self):
        self.perm_selector.reset(save_history=True)
        self.perm_selector.update_prior(self.data[0].unsqueeze(-1))
        self.assertTrue(self.perm_selector.prior_log_inits.logits.allclose(torch.tensor([1 / 3, 2 / 3]).log()))
        self.assertTrue(self.perm_selector.prior_log_current.logits[0, 0].allclose(torch.tensor([2 / 3, 1 / 3]).log()))
        self.assertTrue(self.perm_selector.prior_log_cur_cond_init.logits[0, 0].exp().allclose(self.transition_logits.exp()))
        self.perm_selector.update_prior(self.data[1].unsqueeze(-1))
        self.assertTrue(self.perm_selector.prior_log_cur_cond_init.logits[0, 0].exp().allclose(torch.tensor([[1, ZERO], [2 / 3, 1 / 3]])))
        self.assertTrue(self.perm_selector.prior_log_current.logits[0, 0].allclose(torch.tensor([3 / 4, 1 / 4]).log()))
        self.assertTrue(self.perm_selector.prior_log_inits.logits.allclose(torch.tensor([1 / 4, 3 / 4]).log()))
        self.perm_selector.update_prior(self.data[2].unsqueeze(-1))
        self.assertTrue(self.perm_selector.prior_log_cur_cond_init.logits[0, 0].exp().allclose(torch.tensor([[1, ZERO], [1, ZERO]]), atol=1e-07))
        self.assertTrue(self.perm_selector.prior_log_current.logits[0, 0].exp().allclose(torch.tensor([1, ZERO]), atol=1e-07))
        self.assertTrue(self.perm_selector.prior_log_inits.logits.exp().allclose(torch.tensor([1 / 3, 2 / 3]), atol=1e-07))

    def test_with_perm(self):
        perms = torch.tensor([[0, 1], [1, 0], [0, 1], [0, 1]], dtype=int)
        self.perm_selector.reset(save_history=True)
        self.perm_selector.update_prior(self.data[0].unsqueeze(-1))
        self.assertTrue(self.perm_selector.prior_log_inits.logits.allclose(torch.tensor([1 / 3, 2 / 3]).log()))
        self.assertTrue(self.perm_selector.prior_log_current.logits[0, 1].allclose(torch.tensor([5 / 6, 1 / 6]).log()))
        self.assertTrue(self.perm_selector.prior_log_cur_cond_init.logits[0, 1].exp().allclose(self.transition_logits[perms[1]].exp()))
        self.perm_selector._perm_history.append(perms[1])
        self.perm_selector.update_prior(self.data[1].unsqueeze(-1))
        self.assertTrue(self.perm_selector.prior_log_inits.logits.allclose(torch.tensor([3 / 7, 4 / 7]).log()))
        self.assertTrue(self.perm_selector.prior_log_current.logits[0, 0].allclose(torch.tensor([6 / 7, 1 / 7]).log()))
        self.assertTrue(self.perm_selector.prior_log_cur_cond_init.logits[0, 0].exp().allclose(torch.tensor([[2 / 3, 1 / 3], [1 - ZERO, ZERO]])))
        self.perm_selector._perm_history.append(perms[2])
        self.perm_selector.update_prior(self.data[2].unsqueeze(-1))
        self.assertTrue(self.perm_selector.prior_log_inits.logits.allclose(torch.tensor([1 / 3, 2 / 3]).log()))
        self.assertTrue(self.perm_selector.prior_log_current.logits[0, 0].exp().allclose(torch.tensor([1 - ZERO, ZERO]), atol=1e-07))
        self.assertTrue(self.perm_selector.prior_log_cur_cond_init.logits[0, 0].exp().allclose(torch.tensor([[1 - ZERO, ZERO], [1 - ZERO, ZERO]]), atol=1e-07))

    def test_with_all_vals(self):
        perms = torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1]], dtype=int)
        n_outcomes = 2
        num_perms = len(self.possible_perms)
        num_states = self.initial_logits.shape[0]
        self.perm_selector.reset(save_history=False)
        plyd, plid = self.perm_selector.full_posterior()
        self.assertEqual(plyd.shape, torch.Size([n_outcomes, num_perms]))
        self.assertEqual(plid.shape, torch.Size([n_outcomes, num_perms, num_states]))
        self.assertTrue(plid[:, 0, :].exp().allclose(torch.tensor([[1-ZERO, ZERO], [1/3, 2/3]]), atol=3e-07))
        # self.perm_selector._perm_history.append(perms[0])
        self.perm_selector.update_prior(self.data[0].unsqueeze(-1))
        plyd, plid = self.perm_selector.full_posterior()
        t1 = plid[:, 0, 0, :].exp()
        t2 = torch.tensor([[1/2, 1/2], [1/4, 3/4]])
        self.assertTrue(t1.allclose(t2, atol=3e-07))
        self.perm_selector._perm_history.append(perms[1])
        self.perm_selector.update_prior(self.data[1].unsqueeze(-1))
        self.assertTrue(self.perm_selector.prior_log_cur_cond_init.logits[0, 1].exp().allclose(torch.tensor(
            [[1/2, 1/2],
             [5/6, 1/6]]
        ), atol=3e-07))
        self.assertTrue(self.perm_selector.prior_log_current.logits[0, 1].exp().allclose(torch.tensor(
            [3/4, 1/4]
        ), atol=3e-07))

    def test_entropies(self):
        perms = torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1]], dtype=int)
        self.perm_selector.reset(save_history=False)
        ent = self.perm_selector.expected_entropy()
        should_ent = 3/4*bin_ent(torch.tensor([2/3]).log())
        self.assertTrue(ent[0].allclose(should_ent))
        self.perm_selector.update_prior(self.data[0].unsqueeze(-1))
        ent = self.perm_selector.expected_entropy()
        should_ent = 5/12*bin_ent(torch.tensor([1/5]).log()) + 7/12*bin_ent(torch.tensor([3/7]).log())
        self.assertTrue(ent[0, 1].allclose(should_ent))
        should_ent = 1/3*bin_ent(torch.tensor([1/2]).log()) + 2/3*bin_ent(torch.tensor([1/4]).log())
        self.assertTrue(ent[0, 0].allclose(should_ent))

    def test_reset(self):
        copy_selector = deepcopy(self.perm_selector)
        num_states = self.initial_logits.shape[0]
        idperm = torch.arange(0, num_states, dtype=int)
        for datum in self.data:
            self.perm_selector.update_prior(datum.unsqueeze(-1))
        self.perm_selector.reset()
        self.assertTrue(torch.all(self.perm_selector.prior_log_inits.logits.eq(copy_selector.prior_log_inits.logits)))
        self.assertTrue(torch.all(self.perm_selector.prior_log_current.logits.eq(copy_selector.prior_log_current.logits)))
        self.assertTrue(torch.all(self.perm_selector.prior_log_cur_cond_init.logits.eq(copy_selector.prior_log_cur_cond_init.logits)))

    def test_get_perms(self):
        idperm = torch.arange(2, dtype=int)
        nontriv = torch.tensor([1, 0], dtype=int)
        self.perm_selector.reset(save_history=True)
        perms = self.perm_selector.get_perms(self.data, time_dim=-1)
        hist = self.perm_selector.calc_history
        dists = hist[b"dist_array"]
        entropies = hist[b"entropy_array"]
        self.perm_selector.reset()
        self.perm_selector.update_prior(self.data[0].unsqueeze(-1))
        self.assertTrue(entropies[0].allclose(torch.tensor([1/3*bin_ent(torch.tensor([1/2]).log()) + 2/3*bin_ent(torch.tensor([1/4]).log())], dtype=torch.float)))
        self.assertTrue(perms[0].eq(idperm).all())
        self.perm_selector._perm_history.append(perms[0])
        self.perm_selector.update_prior(self.data[1].unsqueeze(-1))
        entropy = self.perm_selector.expected_entropy()
        self.assertTrue(self.perm_selector.to_perm_index(perms)[1].eq(torch.tensor([entropy.argmin()], dtype=int)).all())
        self.assertTrue(entropies[1].eq(torch.tensor([entropy.min()], dtype=torch.float)).all())
        self.assertTrue(dists[1].eq(self.perm_selector.prior_log_inits.logits).all())
        self.perm_selector._perm_history.append(perms[1])
        self.perm_selector.update_prior(self.data[2].unsqueeze(-1))
        entropy = self.perm_selector.expected_entropy()
        self.assertTrue(self.perm_selector.to_perm_index(perms)[2].eq(torch.tensor([entropy.argmin()], dtype=int)).all())
        self.assertTrue(entropies[2].eq(torch.tensor([entropy.min()], dtype=torch.float)).all())
        self.assertTrue(dists[2].eq(self.perm_selector.prior_log_inits.logits).all())

    def test_single_perm(self):
        n_states = 4
        dirichlet = dist.Dirichlet(torch.ones(n_states) / n_states)
        initial_logits = (torch.ones(n_states) / n_states).log()
        transition_logits = dirichlet.sample((n_states,))
        observation_dist = dist.Bernoulli(torch.rand(n_states))
        possible_perms = torch.arange(n_states)[None, :]
        pdh = PermutedDiscreteHMM(initial_logits, transition_logits, observation_dist)
        perm_selector = MinEntropySelector(possible_perms, pdh,
                                           save_history=True)
        self.assertTrue(perm_selector.expected_entropy().shape == (1,))

if __name__ == '__main__':
    unittest.main()
