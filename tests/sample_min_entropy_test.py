import unittest
from perm_hmm.models.hmms import PermutedDiscreteHMM
import torch
import pyro
import pyro.distributions as dist
from perm_hmm.util import ZERO
from perm_hmm.strategies.min_ent import MinEntropySelector


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
                                         self.observation_dist)
        self.perm_selector = MinEntropySelector(self.possible_perms, self.bdhmm, calibrated=True, save_history=True)

    def test_sample_minent(self):
        x, y = self.bdhmm.sample((10, 7), self.perm_selector)
        perm_array = self.perm_selector.perm_history
        hist = self.perm_selector.calc_history
        dist_array = hist[b"dist_array"]
        entropy_array = hist[b"entropy_array"]
        self.assertTrue(x.shape == (10, 7))
        self.assertTrue(y.shape == (10, 7))
        self.assertTrue(perm_array.shape == (10, 7, 2))
        self.assertTrue(dist_array.shape == (10, 7, 2))
        self.assertTrue(entropy_array.shape == (10, 7))

        b_perm_array = self.perm_selector.get_perms(y, -1, save_history=True)
        b_hist = self.perm_selector.calc_history
        b_dist_array = b_hist[b"dist_array"]
        b_entropy_array = b_hist[b"entropy_array"]
        self.assertTrue(b_dist_array.exp().allclose(dist_array.exp(), atol=1e-6))
        self.assertTrue(torch.all(b_perm_array == perm_array))
        self.assertTrue(b_entropy_array.allclose(entropy_array, atol=1e-7))


    def test_shapes(self):
        shapes = [(10, 7), (), (1,), (10, 1), (1, 1), (10,)]
        for shape in shapes:
            with self.subTest(shape=shape):
                self.perm_selector.reset(save_history=True)
                x, y = self.bdhmm.sample(shape, self.perm_selector)
                print("input shape", shape)
                print("perm_selector shape", self.perm_selector.shape)
                perm_array = self.perm_selector.perm_history
                hist = self.perm_selector.calc_history
                dist_array = hist[b"dist_array"]
                entropy_array = hist[b"entropy_array"]
                if shape == ():
                    shape = (1,)
                self.assertTrue(x.shape == shape)
                self.assertTrue(y.shape == shape)
                self.assertTrue(perm_array.shape == shape + (2,))
                self.assertTrue(dist_array.shape == shape + (2,))
                self.assertTrue(entropy_array.shape == shape)

if __name__ == '__main__':
    unittest.main()
