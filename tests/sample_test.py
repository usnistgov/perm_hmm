import unittest
import torch
import numpy as np
import pyro.distributions as dist
from pyro.distributions import DiscreteHMM
from perm_hmm.models.hmms import SampleableDiscreteHMM, PermutedDiscreteHMM
from perm_hmm.util import ZERO, num_to_data


def to_base(x, y, max_length=None):
    ret = []
    while x > 0:
        ret.append(x % y)
        x = x // y
    if max_length is not None:
        ret += [0]*(max_length-len(ret))
    return list(reversed(ret))


def joint_lp(states, observations, hmm: SampleableDiscreteHMM):
    return hmm.initial_logits[states[..., 0]] + hmm.transition_logits[states[..., :-1], states[..., 1:]].sum(-1) + \
        type(hmm.observation_dist)(hmm.observation_dist._param[states]).log_prob(observations).sum(-1)



class MyTestCase(unittest.TestCase):
    def setUp(self):
        self.num_states = 2
        self.observation_probs = torch.tensor([.5, 1])
        self.observation_dist = dist.Bernoulli(self.observation_probs)
        self.n_outcomes = 2
        self.possible_perms = torch.tensor([[0, 1],
                                            [1, 0]], dtype=int)
        self.num_perms = len(self.possible_perms)
        self.transition_logits = torch.tensor([[1-ZERO, ZERO], [.5, .5]]).log().float()
        self.initial_logits = torch.tensor([.5, .5]).log()
        self.shmm = SampleableDiscreteHMM(self.initial_logits,
                                          self.transition_logits,
                                          self.observation_dist)
        self.normal_shmm = DiscreteHMM(self.initial_logits, self.transition_logits, self.observation_dist)
        self.bhmm = PermutedDiscreteHMM(self.initial_logits,
                                        self.transition_logits,
                                        self.observation_dist,
                                        self.possible_perms)
        self.data = torch.tensor([1.0, 1, 0])
        self.data_2 = torch.tensor([1.0, 1, 1])
        self.data_3 = torch.tensor([0.0, 1, 1])
        self.aye = (torch.eye(2) + ZERO).log()
        self.aye -= self.aye.logsumexp(-1, keepdim=True)
        self.hmm = SampleableDiscreteHMM(self.initial_logits, self.aye,
                                         self.observation_dist)
        self.normal_hmm = DiscreteHMM(self.initial_logits, self.transition_logits, self.observation_dist)

    def test_sample(self):
        num_states = 5
        p = dist.Categorical(torch.eye(5))
        il = \
            dist.Dirichlet(
                torch.ones(num_states, dtype=torch.float)/num_states).sample()
        il = il.log()
        il -= il.logsumexp(-1)
        lm = torch.tensor([
            [.5, .5, ZERO, ZERO, ZERO],
            [ZERO, .5, .5, ZERO, ZERO],
            [ZERO, ZERO, .5, .5, ZERO],
            [ZERO, ZERO, ZERO, .5, .5],
            [ZERO, ZERO, ZERO, ZERO, 1-ZERO],
        ]).log()
        lm -= lm.logsumexp(-1, keepdims=True)
        hmm = SampleableDiscreteHMM(il, lm, p)
        samp = hmm.sample()
        self.assertTrue(samp.observations.shape == ())
        samp = hmm.sample((1,))
        self.assertTrue(samp.observations.shape == (1,))
        samp = hmm.sample((3,))
        self.assertTrue(samp.observations.shape == (3,))
        samp = hmm.sample((500, 4, 8))
        self.assertTrue(samp.observations.shape == (500, 4, 8))
        diffs = np.diff(np.array(samp.states))
        self.assertTrue(np.all((diffs == 0) | (diffs == 1)))
        self.assertTrue((samp.observations.int() == samp.states).all())
        lm0 = torch.tensor([
            [ZERO, 1-ZERO, ZERO, ZERO, ZERO],
            [ZERO, 1-ZERO, ZERO, ZERO, ZERO],
            [ZERO, ZERO, 1-ZERO, ZERO, ZERO],
            [ZERO, ZERO, ZERO, 1-ZERO, ZERO],
            [ZERO, ZERO, ZERO, ZERO, 1-ZERO],
        ]).log()
        lm0 -= lm0.logsumexp(-1, keepdims=True)
        tlm = torch.stack([torch.roll(lm0, (i, i), (0, 1)) for i in range(4)])
        hmm2 = SampleableDiscreteHMM(il, tlm, p)

        samp = hmm2.sample()
        self.assertTrue(samp.observations.shape == (4,))
        samp = hmm2.sample((1,))
        self.assertTrue(samp.observations.shape == (1, 4))
        samp = hmm2.sample((3,))
        self.assertTrue(samp.observations.shape == (3, 4))
        samp = hmm2.sample((500, 2, 8))
        self.assertTrue(samp.observations.shape == (500, 2, 8, 4))
        diffs = np.diff(np.array(samp.states))
        self.assertTrue(np.all((diffs == 0) | (diffs == 1)))
        self.assertTrue((samp.observations.int() == samp.states).all())

    def test_log_prob(self):
        num_states = 3
        d = dist.Dirichlet(torch.ones(num_states)/num_states)
        m = d.sample((num_states,))
        lm = m.log()
        b = dist.Bernoulli(torch.rand(num_states))
        il = dist.Dirichlet(torch.ones(num_states)/num_states).sample()
        hmm = SampleableDiscreteHMM(il, lm, b)
        for i in range(1, 10):
            all_data = torch.tensor([list(map(int, ("{{0:0{}b}}".format(i)).format(j))) for j in range(2 ** i)], dtype=torch.float)
            hlp = hmm.log_prob(all_data)
            ehlp = hlp.logsumexp(-1)
            self.assertTrue(ehlp.allclose(torch.tensor(0.0, dtype=torch.float), atol=1e-6))
        st = 5
        tm = 4
        mm = 100
        dir = dist.Dirichlet(torch.full((st,), 1. / st))
        bern = dist.Bernoulli(torch.rand((st,)))
        il = dir.sample().log()
        tl = dir.sample((st,)).log()
        hmm = SampleableDiscreteHMM(il, tl, bern)

        all_states = torch.stack(
            [torch.tensor(to_base(x, st, tm)) for x in range(st ** tm)])
        all_runs = torch.stack(
            [torch.tensor(num_to_data(x, tm)) for x in range(2 ** tm)]).float()

        s, r = torch.broadcast_tensors(all_states.unsqueeze(-2),
                                       all_runs.unsqueeze(-3))
        x = joint_lp(s, r, hmm)
        self.assertTrue(x.logsumexp(-2).allclose(hmm.log_prob(all_runs)))
        slp = self.shmm.log_prob(self.data[0].unsqueeze(-1))
        hlp = self.hmm.log_prob(self.data[0].unsqueeze(-1))
        self.assertTrue(slp.allclose(hlp))
        slp = self.shmm.log_prob(self.data)
        hlp = self.hmm.log_prob(self.data)
        self.assertTrue(slp.allclose(torch.tensor([3/16]).log()))
        self.assertTrue(hlp.allclose(torch.tensor([1/16]).log()))


if __name__ == '__main__':
    unittest.main()
