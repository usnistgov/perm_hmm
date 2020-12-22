import unittest
import torch
import pyro.distributions as dist
from perm_hmm.classifiers.interrupted import InterruptedClassifier
from perm_hmm.models.hmms import SampleableDiscreteHMM, PermutedDiscreteHMM
from perm_hmm.simulations.simulator import Simulator
from perm_hmm.util import transpositions


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
        self.num_bins = 6

        self.hmm = SampleableDiscreteHMM(
            self.initial_logits,
            self.transition_logits,
            self.observation_dist,
        )
        self.bhmm = PermutedDiscreteHMM.from_hmm(self.hmm, self.possible_perms)
        self.ic = InterruptedClassifier(self.observation_dist, self.testing_states)
        self.bs = Simulator(
            self.bhmm,
            self.testing_states,
            self.num_bins,
        )

    def test_something(self):
        num_samples = 2000
        num_train = 1000
        _ = self.bs.train_ic(num_train)
        x = self.bs.empirical_simulation(num_samples)
        print(x.interrupted_postprocessor.misclassification_rates())
        print(x.naive_postprocessor.misclassification_rates())
        print(x.bayes_postprocessor.misclassification_rates())
        _ = self.bs.exact_train_ic()
        x = self.bs.exact_simulation()
        print(x.interrupted_postprocessor.misclassification_rates())
        print(x.naive_postprocessor.misclassification_rates())
        print(x.bayes_postprocessor.misclassification_rates())


if __name__ == '__main__':
    unittest.main()
