import unittest
import torch
import pyro.distributions as dist
from perm_hmm.classifiers.interrupted import IIDInterruptedClassifier
from perm_hmm.models.hmms import DiscreteHMM, PermutedDiscreteHMM
from perm_hmm.simulator import HMMSimulator
from perm_hmm.util import transpositions, num_to_data
from perm_hmm.policies.min_tree import MinEntPolicy
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
        self.num_steps = 6

        self.hmm = DiscreteHMM(
            self.initial_logits,
            self.transition_logits,
            self.observation_dist,
        )
        self.bhmm = PermutedDiscreteHMM.from_hmm(self.hmm)
        self.perm_policy = MinEntPolicy(
            self.possible_perms,
            self.bhmm,
            save_history=True,
        )
        self.ic = IIDInterruptedClassifier(self.observation_dist, torch.tensor(1.))
        self.bs = HMMSimulator(
            self.bhmm,
        )

    def test_bernoulli(self):
        num_samples = 2000
        num_train = 1000
        x, y = self.hmm.sample((num_train, self.num_steps))
        _ = train_ic(self.ic, y, x[..., 0], self.initial_logits.shape[-1])
        iep = self.bs.simulate(self.num_steps, num_samples)
        x, training_data = self.bhmm.sample((num_train, self.num_steps))
        _ = train_ic(self.ic, training_data, x[..., 0],
                     len(self.bhmm.initial_logits))
        pp = self.bs.simulate(self.num_steps, num_samples, perm_policy=self.perm_policy)
        nop, d = self.bs.simulate(self.num_steps, num_samples, verbosity=1)
        i_classifications = self.ic.classify(d[b"data"], verbosity=0)
        ip = EmpiricalPostprocessor(nop.ground_truth, i_classifications)
        print(ip.confusion_matrix())
        print(nop.confusion_matrix())
        print(pp.confusion_matrix())
        base = len(self.bhmm.observation_dist.enumerate_support())
        data = torch.stack(
            [num_to_data(num, self.num_steps, base) for num in
             range(base ** self.num_steps)]
        ).float()
        lp = self.bhmm.log_prob(data)
        plisd = self.bhmm.posterior_log_initial_state_dist(data)
        log_joint = plisd.T + lp
        _ = exact_train_ic(self.ic, data, log_joint)
        nop = self.bs.all_classifications(self.num_steps)
        pp = self.bs.all_classifications(self.num_steps, perm_policy=self.perm_policy)
        ic_classifications = self.ic.classify(data)
        ip = ExactPostprocessor(log_joint, ic_classifications)
        print(ip.log_misclassification_rate())
        print(nop.log_misclassification_rate())
        print(pp.log_misclassification_rate())


if __name__ == '__main__':
    unittest.main()
