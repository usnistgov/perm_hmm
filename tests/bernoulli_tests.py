import unittest
import torch
import pyro.distributions as dist
from perm_hmm.classifiers.interrupted import IIDInterruptedClassifier
from perm_hmm.models.hmms import DiscreteHMM, PermutedDiscreteHMM
from perm_hmm.simulator import HMMSimulator
from perm_hmm.util import transpositions, num_to_data
from perm_hmm.strategies.min_ent import MinEntropySelector
from perm_hmm.training.interrupted_training import exact_train_ic, train_ic
from perm_hmm.postprocessing import ExactPostprocessor, EmpiricalPostprocessor


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

        self.hmm = DiscreteHMM(
            self.initial_logits,
            self.transition_logits,
            self.observation_dist,
        )
        self.bhmm = PermutedDiscreteHMM.from_hmm(self.hmm)
        self.perm_selector = MinEntropySelector(self.possible_perms, self.bhmm,
                                                save_history=True)
        self.ic = IIDInterruptedClassifier(self.observation_dist, self.testing_states)
        self.bs = HMMSimulator(
            self.bhmm,
        )

    def test_something(self):
        num_samples = 2000
        num_train = 1000
        x, y = self.hmm.sample((num_train, self.num_bins))
        _ = train_ic(self.ic, self.testing_states, y, x[..., 0], self.initial_logits.shape[-1])
        iep = self.bs.simulate(self.num_bins, num_samples, self.testing_states)
        x, training_data = self.bhmm.sample((num_train, self.num_bins))
        _ = train_ic(self.ic, self.testing_states, training_data, x[..., 0],
                     len(self.bhmm.initial_logits))
        pp = self.bs.simulate(self.num_bins, num_samples, self.testing_states,
                                perm_selector=self.perm_selector)
        nop, d = self.bs.simulate(self.num_bins, num_samples, self.testing_states, verbosity=1)
        i_classifications = self.ic.classify(d[b"data"], self.testing_states, verbosity=0)
        ip = EmpiricalPostprocessor(nop.ground_truth, self.testing_states, i_classifications)
        i_classifications = ip.classifications
        no_classifications = nop.classifications
        p_classifications = pp.classifications
        print(ip.misclassification_rates())
        print(nop.misclassification_rates())
        print(pp.misclassification_rates())
        base = len(self.bhmm.observation_dist.enumerate_support())
        data = torch.stack(
            [num_to_data(num, self.num_bins, base) for num in
             range(base ** self.num_bins)]
        ).float()
        lp = self.bhmm.log_prob(data)
        plisd = self.bhmm.posterior_log_initial_state_dist(data)
        log_joint = plisd.T + lp
        _ = exact_train_ic(self.ic, self.testing_states, data, log_joint)
        nop = self.bs.all_classifications(self.num_bins, self.testing_states)
        pp = self.bs.all_classifications(self.num_bins, self.testing_states, perm_selector=self.perm_selector)
        ic_classifications = self.ic.classify(data, self.testing_states)
        ip = ExactPostprocessor(log_joint, self.testing_states, ic_classifications)
        i_classifications = ip.classifications
        no_classifications = nop.classifications
        p_classifications = pp.classifications
        print(ip.log_misclassification_rate())
        print(nop.log_misclassification_rate())
        print(pp.log_misclassification_rate())


if __name__ == '__main__':
    unittest.main()
