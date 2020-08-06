import unittest
import sys
import torch
import pyro.distributions as dist
import pyro
from bayes_perm_hmm.min_entropy_hmm import PermutedDiscreteHMM
from bayes_perm_hmm.interrupted import InterruptedClassifier
from bayes_perm_hmm.sampleable import SampleableDiscreteHMM
from bayes_perm_hmm.postprocessing import InterruptedEmpiricalPostprocessor, InterruptedExactPostprocessor
import bayes_perm_hmm.training
from bayes_perm_hmm.util import ZERO, transpositions, num_to_data


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.num_states = 5
        dir = dist.Dirichlet(torch.ones(self.num_states)/self.num_states)
        self.observation_dist = dist.Bernoulli(torch.rand((self.num_states,)))
        self.transition_logits = dir.sample((self.num_states,)).log()
        self.initial_logits = dir.sample().log()
        self.num_testing_states = 3
        self.testing_states = torch.multinomial(dir.sample(), self.num_testing_states)
        while (self.initial_logits.exp()[self.testing_states] < .1).any():
            self.testing_states = torch.multinomial(dir.sample(), self.num_testing_states)
            self.initial_logits = dir.sample().log()
        self.possible_perms = \
            torch.stack(
                [torch.arange(self.num_states)] +
                transpositions(self.num_states)
            )

        self.hmm = SampleableDiscreteHMM(
            self.initial_logits,
            self.transition_logits,
            self.observation_dist,
        )
        self.bhmm = PermutedDiscreteHMM.from_hmm(self.hmm, self.possible_perms)
        self.ic = InterruptedClassifier(self.observation_dist, self.testing_states)

    def test_ic(self):
        num_training_samples = 100
        time_dim = 6
        training_data = self.hmm.sample((num_training_samples, time_dim))
        ground_truth = training_data.states[..., 0]
        while (~((ground_truth.unsqueeze(-1) == self.testing_states.unsqueeze(-2)).any(-2))).any(-1):
            training_data = self.hmm.sample((num_training_samples, time_dim))
            ground_truth = training_data.states[..., 0]
        _ = bayes_perm_hmm.training.train(self.ic, training_data.observations, ground_truth, self.num_states)
        num_testing_samples = 300
        testing_data = self.hmm.sample((num_testing_samples, time_dim))
        class_break_ratio = self.ic.classify(testing_data.observations)
        iep = InterruptedEmpiricalPostprocessor(
            testing_data.states[..., 0],
            self.testing_states,
            self.num_states,
            *class_break_ratio,
        )
        res = iep.misclassification_rates()
        print(res)
        self.assertTrue(res.confusions.rate.sum(-1)[self.testing_states].allclose(torch.tensor(1.)))
        all_possible_runs = torch.stack([num_to_data(x, time_dim) for x in range(2**time_dim)])
        plisd = self.hmm.posterior_log_initial_state_dist(all_possible_runs)
        lp = self.hmm.log_prob(all_possible_runs)
        class_break_ratio = self.ic.classify(all_possible_runs)
        iep = InterruptedExactPostprocessor(
            lp,
            plisd,
            self.initial_logits,
            self.testing_states,
            class_break_ratio,
        )
        res = iep.misclassification_rates()
        self.assertTrue(res.confusions.sum(-1)[self.testing_states].allclose(torch.tensor(1.)))
        print(res)

if __name__ == '__main__':
    unittest.main()
