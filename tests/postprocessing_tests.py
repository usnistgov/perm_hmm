import unittest
import torch
import torch.distributions
import pyro.distributions as dist
import perm_hmm.models.hmms
from perm_hmm.strategies.min_ent import MinEntropySelector
from perm_hmm.models.hmms import DiscreteHMM, PermutedDiscreteHMM
from perm_hmm.classifiers.interrupted import IIDInterruptedClassifier
from perm_hmm.training.interrupted_training import train_ic, exact_train_ic
from perm_hmm.postprocessing import ExactPostprocessor, EmpiricalPostprocessor
from perm_hmm.util import transpositions, num_to_data, ZERO
from perm_hmm.classifiers.generic_classifiers import MAPClassifier

class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.num_states = 5
        self.num_testing_states = 3
        ts = torch.randint(self.num_states, (self.num_testing_states,))
        while len(ts.unique()) != len(ts):
            ts = torch.randint(self.num_states, (self.num_testing_states,))
        self.testing_states = ts
        dir = dist.Dirichlet(torch.ones(self.num_testing_states)/self.num_testing_states)
        not_states = torch.tensor(list(set(range(self.num_states)).difference(set(self.testing_states.tolist()))))
        il = dir.sample()
        while (il < 1e-1).any():
            il = dir.sample()
        self.initial_logits = torch.empty(self.num_states)
        self.initial_logits[self.testing_states] = il
        self.initial_logits[not_states] = ZERO
        self.initial_logits = self.initial_logits.log()
        dir = dist.Dirichlet(torch.ones(self.num_states)/self.num_states)
        self.transition_logits = dir.sample((self.num_states,)).log()
        self.observation_probs = torch.rand((self.num_states,))
        self.observation_dist = dist.Bernoulli(self.observation_probs)
        self.possible_perms = torch.stack([torch.arange(self.num_states)] + transpositions(self.num_states))
        self.bdhmm = PermutedDiscreteHMM(self.initial_logits,
                                         self.transition_logits,
                                         self.observation_dist)
        self.perm_selector = MinEntropySelector(self.possible_perms, self.bdhmm,
                                                save_history=True)
        self.shmm = DiscreteHMM(self.initial_logits,
                                self.transition_logits,
                                self.observation_dist)

    def test_something(self):
        max_t = 10
        num_samples = 1000
        x, y = self.shmm.sample((num_samples, max_t))
        ground_truth = x[..., 0]
        log_post_dist = self.shmm.posterior_log_initial_state_dist(y)
        classifications = log_post_dist.argmax(-1)
        ep = EmpiricalPostprocessor(ground_truth, classifications)
        mr = ep.confusion_matrix(.95)
        print(mr)
        self.assertTrue(torch.allclose(mr[b"matrix"][self.testing_states].sum(-1), torch.tensor(1.)))
        self.assertTrue((mr[b"lower"][self.testing_states] <= mr[b"matrix"][self.testing_states]).all())
        self.assertTrue((mr[b"upper"][self.testing_states] >= mr[b"matrix"][self.testing_states]).all())
        v = self.perm_selector.get_perms(y, -1)
        hist = self.perm_selector.calc_history
        b_log_post_dist = hist[b"dist_array"][..., -1, :]
        b_classifications = b_log_post_dist.argmax(-1)
        bep = EmpiricalPostprocessor(ground_truth, b_classifications)
        mr = bep.confusion_matrix()
        print(mr)
        self.assertTrue(torch.allclose(mr[b"matrix"][self.testing_states].sum(-1), torch.tensor(1.)))
        self.assertTrue((mr[b"lower"][self.testing_states] <= mr[b"matrix"][self.testing_states]).all())
        self.assertTrue((mr[b"upper"][self.testing_states] >= mr[b"matrix"][self.testing_states]).all())

        observation_params = self.observation_dist._param
        bright_state = observation_params.argmax(-1)
        dark_state = observation_params.argmin(-1)
        testing_states = torch.tensor([dark_state, bright_state])
        ic = IIDInterruptedClassifier(
            dist.Bernoulli(self.observation_probs[testing_states]),
            torch.tensor(1.),
            testing_states=testing_states,
        )
        train_x, train_y = self.shmm.sample((num_samples, max_t))
        ground_truth = train_x[..., 0]
        _ = train_ic(ic, train_y, train_x[..., 0])
        ic_results = ic.classify(y)
        ip = EmpiricalPostprocessor(
            ground_truth,
            ic_results
        )
        mr = ip.confusion_matrix()
        print(mr)
        self.assertTrue(torch.allclose(mr[b"matrix"][self.testing_states].sum(-1), torch.tensor(1.)))
        self.assertTrue((mr[b"lower"][self.testing_states] <= mr[b"matrix"][self.testing_states]).all())
        self.assertTrue((mr[b"upper"][self.testing_states] >= mr[b"matrix"][self.testing_states]).all())

        all_data = torch.stack([num_to_data(x, max_t) for x in range(2**max_t)])
        all_naive_post = self.shmm.posterior_log_initial_state_dist(all_data)
        naive_lp = self.shmm.log_prob(all_data)
        log_joint = all_naive_post.T + naive_lp
        map_class = MAPClassifier(self.shmm)
        classifications = map_class.classify(all_data)
        np = ExactPostprocessor(
            log_joint,
            classifications,
        )
        mr = np.log_misclassification_rate()
        conf = np.log_confusion_matrix()
        print(mr)
        self.assertTrue(conf[self.testing_states].logsumexp(-1).allclose(torch.tensor(0.), atol=1e-6))
        self.perm_selector.reset(save_history=True)
        bayes_results = self.perm_selector.get_perms(all_data, -1)
        hist = self.perm_selector.calc_history
        phmm = self.bdhmm.expand_with_perm(bayes_results)
        b_map_class = MAPClassifier(phmm)
        lp = phmm.log_prob(all_data)
        plisd = phmm.posterior_log_initial_state_dist(all_data)
        b_log_joint = lp + plisd.T
        b_classifications = b_map_class.classify(all_data)
        bp = ExactPostprocessor(
            b_log_joint,
            b_classifications,
        )
        mr = bp.log_misclassification_rate()
        conf = bp.log_confusion_matrix()
        print(mr)
        self.assertTrue(conf[self.testing_states].logsumexp(-1).allclose(torch.tensor(0.), atol=1e-6))
        _ = exact_train_ic(ic, all_data, log_joint)
        ic_results = ic.classify(all_data)
        ip = ExactPostprocessor(
            log_joint,
            ic_results,
        )
        ave = ip.log_misclassification_rate()
        conf = ip.log_confusion_matrix()
        print(mr)
        self.assertTrue(conf[self.testing_states].logsumexp(-1).allclose(torch.tensor(0.), atol=1e-6))


if __name__ == '__main__':
    unittest.main()
