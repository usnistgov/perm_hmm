import unittest
import sys
from copy import deepcopy
import torch
import pyro.distributions as dist
import pyro
from bayes_perm_hmm.sampleable import SampleableDiscreteHMM
from bayes_perm_hmm.min_entropy_hmm import PermutedDiscreteHMM
from bayes_perm_hmm.util import bin_ent, ZERO


class BdhmmTestCase(unittest.TestCase):
    def setUp(self):
        self.num_states = 2
        self.observation_probs = torch.tensor([.5, 1])
        self.observation_constructor = pyro.distributions.Bernoulli
        self.observation_dist = \
            self.observation_constructor(self.observation_probs)
        self.n_outcomes = 2
        self.possible_outputs = torch.arange(2, dtype=torch.float).unsqueeze(-1)
        self.possible_perms = torch.tensor([[0, 1],
                                            [1, 0]], dtype=int)
        self.num_perms = len(self.possible_perms)
        self.transition_logits = torch.tensor([[1-ZERO, ZERO], [.5, .5]]).log().float()
        self.initial_logits = torch.tensor([.5, .5]).log()
        self.bdhmm = PermutedDiscreteHMM(self.initial_logits,
                                         self.transition_logits,
                                         self.observation_dist,
                                         self.possible_perms)
        self.shmm = SampleableDiscreteHMM(self.initial_logits,
                                          self.transition_logits,
                                          self.observation_dist)
        self.data = torch.tensor([1.0, 1, 0])
        self.data_1 = torch.tensor([1, 1.0, 0, 1, 0, 0])
        self.data_2 = torch.tensor([1.0, 1, 1])
        self.data_3 = torch.tensor([0.0, 1, 1])
        self.integration_time = 1

    def test_posterior_init(self):
        idperms = self.possible_perms[torch.zeros(self.data.shape[0], dtype=int)]
        bpinit = self.bdhmm.get_posterior(self.data, idperms)
        spinit = self.shmm.posterior_log_initial_state_dist(self.data)
        self.assertTrue(bpinit[-1].allclose(spinit))
        idperms = self.possible_perms[torch.zeros(self.data_2.shape[0], dtype=int)]
        bpinit = self.bdhmm.get_posterior(self.data_2, idperms)
        spinit = self.shmm.posterior_log_initial_state_dist(self.data_2)
        self.assertTrue(bpinit[-1].allclose(spinit))
        idperms = self.possible_perms[torch.zeros(self.data_3.shape[0], dtype=int)]
        bpinit = self.bdhmm.get_posterior(self.data_3, idperms)
        spinit = self.shmm.posterior_log_initial_state_dist(self.data_3)
        self.assertTrue(bpinit[-1].allclose(spinit))
        all_data = torch.tensor([list(map(int, ("{{0:0{}b}}".format(6)).format(j))) for j in range(2 ** 6)], dtype=torch.float)
        slp = self.shmm.log_prob(all_data)
        sllp = torch.tensor([self.shmm.log_prob(data) for data in all_data])
        self.assertTrue(slp.allclose(sllp))

    def test_no_perms(self):
        idperm = torch.arange(0, self.num_states, dtype=int)
        self.bdhmm.update_prior(self.data[0], idperm)
        self.assertTrue(self.bdhmm.prior_log_inits.logits.allclose(torch.tensor([1 / 3, 2 / 3]).log()))
        self.assertTrue(self.bdhmm.prior_log_current.logits[0].allclose(torch.tensor([2 / 3, 1 / 3]).log()))
        self.assertTrue(self.bdhmm.prior_log_cur_cond_init.logits[0].exp().allclose(self.transition_logits.exp()))
        self.bdhmm.update_prior(self.data[1], idperm)
        self.assertTrue(self.bdhmm.prior_log_cur_cond_init.logits[0].exp().allclose(torch.tensor([[1, ZERO], [2 / 3, 1 / 3]])))
        self.assertTrue(self.bdhmm.prior_log_current.logits[0].allclose(torch.tensor([3 / 4, 1 / 4]).log()))
        self.assertTrue(self.bdhmm.prior_log_inits.logits.allclose(torch.tensor([1 / 4, 3 / 4]).log()))
        self.bdhmm.update_prior(self.data[2], idperm)
        self.assertTrue(self.bdhmm.prior_log_cur_cond_init.logits[0].exp().allclose(torch.tensor([[1, ZERO], [1, ZERO]]), atol=1e-07))
        self.assertTrue(self.bdhmm.prior_log_current.logits[0].exp().allclose(torch.tensor([1, ZERO]), atol=1e-07))
        self.assertTrue(self.bdhmm.prior_log_inits.logits.exp().allclose(torch.tensor([1 / 3, 2 / 3]), atol=1e-07))

    def test_with_perm(self):
        perms = torch.tensor([[0, 1], [1, 0], [0, 1], [0, 1]], dtype=int)
        self.bdhmm.update_prior(self.data[0], perms[0])
        self.assertTrue(self.bdhmm.prior_log_inits.logits.allclose(torch.tensor([1 / 3, 2 / 3]).log()))
        self.assertTrue(self.bdhmm.prior_log_current.logits[1].allclose(torch.tensor([5 / 6, 1 / 6]).log()))
        self.assertTrue(self.bdhmm.prior_log_cur_cond_init.logits[1].exp().allclose(self.transition_logits[perms[1]].exp()))
        self.bdhmm.update_prior(self.data[1], perms[1])
        self.assertTrue(self.bdhmm.prior_log_inits.logits.allclose(torch.tensor([3 / 7, 4 / 7]).log()))
        self.assertTrue(self.bdhmm.prior_log_current.logits[0].allclose(torch.tensor([6 / 7, 1 / 7]).log()))
        self.assertTrue(self.bdhmm.prior_log_cur_cond_init.logits[0].exp().allclose(torch.tensor([[2 / 3, 1 / 3], [1 - ZERO, ZERO]])))
        self.bdhmm.update_prior(self.data[2], perms[2])
        self.assertTrue(self.bdhmm.prior_log_inits.logits.allclose(torch.tensor([1 / 3, 2 / 3]).log()))
        self.assertTrue(self.bdhmm.prior_log_current.logits[0].exp().allclose(torch.tensor([1 - ZERO, ZERO]), atol=1e-07))
        self.assertTrue(self.bdhmm.prior_log_cur_cond_init.logits[0].exp().allclose(torch.tensor([[1 - ZERO, ZERO], [1 - ZERO, ZERO]]), atol=1e-07))

    def test_with_all_vals(self):
        perms = torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1]], dtype=int)
        plyd, plid = self.bdhmm.full_posterior()
        self.assertEqual(plyd.shape, torch.Size([self.n_outcomes, self.num_perms]))
        self.assertEqual(plid.shape, torch.Size([self.n_outcomes, self.num_perms, self.num_states]))
        self.assertTrue(plid[:, 0, :].exp().allclose(torch.tensor([[1-ZERO, ZERO], [1/3, 2/3]]), atol=3e-07))
        self.bdhmm.update_prior(self.data[0], perms[0])
        plyd, plid = self.bdhmm.full_posterior()
        self.assertTrue(plid[:, 0, :].exp().allclose(torch.tensor([[1/2, 1/2], [1/4, 3/4]]), atol=3e-07))
        self.bdhmm.update_prior(self.data[1], perms[1])
        self.assertTrue(self.bdhmm.prior_log_cur_cond_init.logits[1].exp().allclose(torch.tensor(
            [[1/2, 1/2],
             [5/6, 1/6]]
        ), atol=3e-07))
        self.assertTrue(self.bdhmm.prior_log_current.logits[1].exp().allclose(torch.tensor(
            [3/4, 1/4]
        ), atol=3e-07))

    def test_entropies(self):
        perms = torch.tensor([[0, 1], [0, 1], [1, 0], [0, 1]], dtype=int)
        self.assertTrue(self.bdhmm.expected_entropy()[0].allclose(3/4*bin_ent(torch.tensor([2/3]).log())))
        self.bdhmm.update_prior(self.data[0], perms[0])
        self.assertTrue(self.bdhmm.expected_entropy()[0].allclose(1/3*bin_ent(torch.tensor([1/2]).log()) + 2/3*bin_ent(torch.tensor([1/4]).log())))
        self.assertTrue(self.bdhmm.expected_entropy()[1].allclose(5/12*bin_ent(torch.tensor([1/5]).log()) + 7/12*bin_ent(torch.tensor([3/7]).log())))

    def test_reset(self):
        copyhmm = deepcopy(self.bdhmm)
        idperm = torch.arange(0, self.num_states, dtype=int)
        for datum in self.data:
            self.bdhmm.update_prior(datum, idperm)
        self.bdhmm.reset()
        self.assertTrue(torch.all(self.bdhmm.prior_log_inits.logits.eq(copyhmm.prior_log_inits.logits)))
        self.assertTrue(torch.all(self.bdhmm.prior_log_current.logits.eq(copyhmm.prior_log_current.logits)))
        self.assertTrue(torch.all(self.bdhmm.prior_log_cur_cond_init.logits.eq(copyhmm.prior_log_cur_cond_init.logits)))

    def test_get_perms(self):
        idperm = torch.arange(2, dtype=int)
        nontriv = torch.tensor([1, 0], dtype=int)
        perms, (dists, entropies) = self.bdhmm.get_perms(self.data)
        self.bdhmm.reset()
        self.bdhmm.update_prior(self.data[0], idperm)
        self.assertTrue(entropies[0].allclose(torch.tensor([1/3*bin_ent(torch.tensor([1/2]).log()) + 2/3*bin_ent(torch.tensor([1/4]).log())], dtype=torch.float)))
        self.assertTrue(perms[0].eq(idperm).all())
        self.bdhmm.update_prior(self.data[1], perms[0])
        entropy = self.bdhmm.expected_entropy()
        self.assertTrue(self.bdhmm.to_perm_index(perms)[1].eq(torch.tensor([entropy.argmin()], dtype=int)).all())
        self.assertTrue(entropies[1].eq(torch.tensor([entropy.min()], dtype=torch.float)).all())
        self.assertTrue(dists[1].eq(self.bdhmm.prior_log_inits.logits).all())
        self.bdhmm.update_prior(self.data[2], perms[1])
        entropy = self.bdhmm.expected_entropy()
        self.assertTrue(self.bdhmm.to_perm_index(perms)[2].eq(torch.tensor([entropy.argmin()], dtype=int)).all())
        self.assertTrue(entropies[2].eq(torch.tensor([entropy.min()], dtype=torch.float)).all())
        self.assertTrue(dists[2].eq(self.bdhmm.prior_log_inits.logits).all())

    def test_single_perm(self):
        n_states = 4
        dirichlet = dist.Dirichlet(torch.ones(n_states) / n_states)
        initial_logits = (torch.ones(n_states) / n_states).log()
        transition_logits = dirichlet.sample((n_states,))
        observation_dist = dist.Bernoulli(torch.rand(n_states))
        possible_perms = torch.arange(n_states)[None, :]
        pdh = PermutedDiscreteHMM(initial_logits, transition_logits, observation_dist, possible_perms)

        self.assertTrue(pdh.expected_entropy().shape == (1,))

if __name__ == '__main__':
    unittest.main()
