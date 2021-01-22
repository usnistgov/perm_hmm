import unittest
import numpy as np
import torch
import pyro
import pyro.distributions as dist
from perm_hmm.models.hmms import DiscreteHMM, PermutedDiscreteHMM
from perm_hmm.example_systems.three_states import three_state_params


class MyTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.a = np.exp(-3)
        self.b = np.exp(-4)
        self.initial_probs, self.transition_probs, self.observation_probs = three_state_params(self.a, self.b)
        self.hmm = DiscreteHMM(torch.tensor(self.initial_probs).log(), torch.tensor(self.transition_probs).log(), dist.Categorical(torch.tensor(self.observation_probs)))
        self.phmm = PermutedDiscreteHMM.from_hmm(self.hmm)

    def test_something(self):
        zero_pisd = self.hmm.posterior_log_initial_state_dist(torch.tensor(0.)).exp()
        expected_zero_pisd = torch.tensor([2/3, 1/3, 0])
        self.assertTrue(zero_pisd.allclose(expected_zero_pisd))
        one_pisd = self.hmm.posterior_log_initial_state_dist(torch.tensor(1.)).exp()
        expected_one_pisd = torch.tensor([1/3, 1/3, 1/3])
        self.assertTrue(one_pisd.allclose(expected_one_pisd))
        two_pisd = self.hmm.posterior_log_initial_state_dist(torch.tensor(1.)).exp()
        expected_two_pisd = torch.tensor([0., 1/3, 2/3])
        self.assertTrue(two_pisd.allclose(expected_two_pisd))
        zero_pisd = self.phmm.posterior_log_initial_state_dist(torch.tensor(0.)).exp()
        one_pisd = self.phmm.posterior_log_initial_state_dist(torch.tensor(1.)).exp()
        two_pisd = self.phmm.posterior_log_initial_state_dist(torch.tensor(1.)).exp()
        self.assertTrue(zero_pisd.allclose(expected_zero_pisd))
        self.assertTrue(one_pisd.allclose(expected_one_pisd))
        self.assertTrue(two_pisd.allclose(expected_two_pisd))

        zero_p = self.hmm.log_prob(torch.tensor(0.)).exp()
        expected_zero_p = torch.tensor(1/2*(1-self.a))
        self.assertTrue(zero_p.allclose(expected_zero_p))
        one_p = self.hmm.log_prob(torch.tensor(1.)).exp()
        expected_one_p = torch.tensor(self.a)
        self.assertTrue(one_p.allclose(expected_one_p))
        two_p = self.hmm.log_prob(torch.tensor(1.)).exp()
        expected_two_p = torch.tensor(1/2*(1-self.a))
        self.assertTrue(two_p.allclose(expected_two_p))
        zero_p = self.phmm.log_prob(torch.tensor(0.)).exp()
        one_p = self.phmm.log_prob(torch.tensor(1.)).exp()
        two_p = self.phmm.log_prob(torch.tensor(1.)).exp()
        self.assertTrue(zero_p.allclose(expected_zero_p))
        self.assertTrue(one_p.allclose(expected_one_p))
        self.assertTrue(two_p.allclose(expected_two_p))




if __name__ == '__main__':
    unittest.main()
