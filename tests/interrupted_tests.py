import unittest
import torch
import pyro.distributions as dist
from perm_hmm.classifiers.interrupted import IIDInterruptedClassifier
from perm_hmm.models.hmms import DiscreteHMM, PermutedDiscreteHMM
from perm_hmm.postprocessing import ExactPostprocessor, EmpiricalPostprocessor
import perm_hmm.training.interrupted_training
from perm_hmm.util import transpositions, num_to_data, ZERO
from perm_hmm.strategies.min_ent import MinEntropySelector


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.num_states = 5
        dir = dist.Dirichlet(torch.ones(self.num_states)/self.num_states)
        self.observation_dist = dist.Bernoulli(torch.rand((self.num_states,)))
        self.transition_logits = dir.sample((self.num_states,)).log()
        self.num_testing_states = 3
        self.testing_states = torch.randint(self.num_states, (self.num_testing_states,))
        while len(self.testing_states.unique()) != len(self.testing_states):
            self.testing_states = torch.randint(self.num_states, (self.num_testing_states,))
        dir = dist.Dirichlet(torch.ones(self.num_testing_states)/self.num_testing_states)
        not_states = torch.tensor(list(set(range(self.num_states)).difference(set(self.testing_states.tolist()))))
        il = dir.sample()
        self.initial_logits = torch.empty(self.num_states)
        self.initial_logits[self.testing_states] = il
        self.initial_logits[not_states] = ZERO
        self.initial_logits = self.initial_logits.log()
        self.possible_perms = \
            torch.stack(
                [torch.arange(self.num_states)] +
                transpositions(self.num_states)
            )

        self.hmm = DiscreteHMM(
            self.initial_logits,
            self.transition_logits,
            self.observation_dist,
        )
        self.bhmm = PermutedDiscreteHMM.from_hmm(self.hmm)
        self.perm_selector = MinEntropySelector(self.possible_perms, self.bhmm,
                                                save_history=True)
        self.ic = IIDInterruptedClassifier(self.observation_dist, torch.tensor(1.))

    def test_ic(self):
        num_training_samples = 100
        time_dim = 6
        training_data = self.hmm.sample((num_training_samples, time_dim))
        ground_truth = training_data.states[..., 0]
        _ = perm_hmm.training.interrupted_training.train_ic(self.ic, training_data.observations, ground_truth)
        num_testing_samples = 300
        testing_data = self.hmm.sample((num_testing_samples, time_dim))
        i_class = self.ic.classify(testing_data.observations, verbosity=0)
        iep = EmpiricalPostprocessor(
            testing_data.states[..., 0],
            i_class,
        )
        rate = iep.misclassification_rate()
        conf = iep.confusion_matrix(.95)
        print(conf)
        self.assertTrue(conf[b"rate"][self.testing_states].sum(-1).allclose(torch.tensor(1.)))
        all_possible_runs = torch.stack([num_to_data(x, time_dim) for x in range(2**time_dim)])
        plisd = self.hmm.posterior_log_initial_state_dist(all_possible_runs)
        lp = self.hmm.log_prob(all_possible_runs)
        log_joint = plisd.T + lp
        i_class = self.ic.classify(all_possible_runs, verbosity=0)
        iep = ExactPostprocessor(
            log_joint,
            i_class,
        )
        res = iep.log_misclassification_rate()
        conf = iep.log_confusion_matrix()
        self.assertTrue(conf[self.testing_states].logsumexp(-1).allclose(torch.tensor(0.), atol=1e-5))
        print(res)

if __name__ == '__main__':
    unittest.main()
