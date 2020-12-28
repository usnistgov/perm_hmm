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
        self.deep_hmm = PermutedDiscreteHMM(
            self.initial_logits,
            self.transition_logits,
            dist.Bernoulli(torch.rand(2, 2))
        )
        self.deep_perm_selector = MinEntropySelector(self.possible_perms, self.deep_hmm, calibrated=True, save_history=True)
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

        self.perm_selector.reset(save_history=True)
        b_perm_array = self.perm_selector.get_perms(y, -1)
        b_hist = self.perm_selector.calc_history
        b_dist_array = b_hist[b"dist_array"]
        b_entropy_array = b_hist[b"entropy_array"]
        self.assertTrue(b_dist_array.exp().allclose(dist_array.exp(), atol=1e-6))
        self.assertTrue(torch.all(b_perm_array == perm_array))
        self.assertTrue(b_entropy_array.allclose(entropy_array, atol=1e-7))

    def test_shapes(self):
        shapes = [(10, 7), (), (1,), (10, 1), (1, 1), (10,)]
        # shallow = [(self.perm_selector, self.bdhmm), (self.deep_perm_selector, self.deep_hmm)]
        shallow = [(self.perm_selector, self.bdhmm)]
        for shape in shapes:
            for typ in shallow:
                ps, hmm = typ
                with self.subTest(shape=shape, hmm=hmm):
                    x, y = hmm.sample(shape)
                    t_shape = shape
                    if hmm == self.deep_hmm:
                        t_shape = t_shape + self.deep_hmm.observation_dist.batch_shape[:-1]
                    else:
                        if t_shape == ():
                            t_shape = (1,)
                    self.assertTrue(x.shape == t_shape)
                    self.assertTrue(
                        y.shape == t_shape + hmm.observation_dist.event_shape)
                with self.subTest(shape=shape, ps=ps, hmm=hmm):
                    ps.reset(save_history=True)
                    x, y = hmm.sample(shape, ps)
                    print("input shape", shape)
                    print("ps shape", ps.shape)
                    perm_array = ps.perm_history
                    hist = ps.calc_history
                    dist_array = hist[b"dist_array"]
                    entropy_array = hist[b"entropy_array"]
                    t_shape = shape
                    s_shape = shape
                    if hmm == self.deep_hmm:
                        t_shape = t_shape + self.deep_hmm.observation_dist.batch_shape[:-1]
                        s_shape = s_shape + self.deep_hmm.observation_dist.batch_shape[:-1]
                    else:
                        if t_shape == ():
                            t_shape = (1,)
                            s_shape = ()
                    self.assertTrue(x.shape == t_shape)
                    self.assertTrue(y.shape == t_shape + hmm.observation_dist.event_shape)
                    self.assertTrue(perm_array.shape == t_shape + (2,))
                    self.assertTrue(dist_array.shape == t_shape + (2,))
                    self.assertTrue(entropy_array.shape == t_shape)


if __name__ == '__main__':
    unittest.main()
