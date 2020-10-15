import unittest
from bayes_perm_hmm.min_entropy_hmm import PermutedDiscreteHMM
import torch
import pyro
import pyro.distributions as dist
from bayes_perm_hmm.util import ZERO


class MyTestCase(unittest.TestCase):
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

    def test_sample_minent(self):
        x, y, perm_array, (dist_array, entropy_array) = \
            self.bdhmm.sample_min_entropy((10, 7))
        self.assertTrue(x.shape == (10, 7))
        self.assertTrue(y.shape == (10, 7))
        self.assertTrue(perm_array.shape == (10, 7, 2))
        self.assertTrue(dist_array.shape == (10, 7, 2))
        self.assertTrue(entropy_array.shape == (10, 7))

        b_perm_array, (b_dist_array, b_entropy_array) = \
            self.bdhmm.get_perms(y)
        self.assertTrue(b_dist_array.exp().allclose(dist_array.exp(), atol=1e-6))
        self.assertTrue(torch.all(b_perm_array == perm_array))
        self.assertTrue(b_entropy_array.allclose(entropy_array, atol=1e-7))

        x, y, perm_array, (dist_array, entropy_array) = \
            self.bdhmm.sample_min_entropy()
        self.assertTrue(y.shape == ())
        x, y, perm_array, (dist_array, entropy_array) = \
            self.bdhmm.sample_min_entropy((1,))
        self.assertTrue(y.shape == (1,))
        x, y, perm_array, (dist_array, entropy_array) = \
            self.bdhmm.sample_min_entropy((10, 1))
        self.assertTrue(y.shape == (10, 1))

if __name__ == '__main__':
    unittest.main()
