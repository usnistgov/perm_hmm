import unittest
import torch
import torch.distributions
import pyro
import pyro.distributions as dist
from bayes_perm_hmm.sampleable import SampleableDiscreteHMM
from bayes_perm_hmm.min_entropy_hmm import PermutedDiscreteHMM
from bayes_perm_hmm.interrupted import InterruptedClassifier
from bayes_perm_hmm.training import train, exact_train
import bayes_perm_hmm.postprocessing
from bayes_perm_hmm.util import ZERO, transpositions, num_to_data


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.num_states = 5
        self.num_testing_states = 3
        dir = dist.Dirichlet(torch.ones(self.num_states)/self.num_states)
        self.initial_logits = dir.sample().log()
        self.testing_states = torch.multinomial(torch.ones(self.num_states)/self.num_states, self.num_testing_states)
        while (self.initial_logits[self.testing_states].exp() < 1e-2).any():
            self.initial_logits = dir.sample().log()
            self.testing_states = torch.multinomial(
                torch.ones(self.num_states) / self.num_states,
                self.num_testing_states)
        self.transition_logits = dir.sample((self.num_states,)).log()
        self.observation_dist = dist.Bernoulli(torch.rand((self.num_states,)))
        self.possible_perms = torch.stack([torch.arange(self.num_states)] + transpositions(self.num_states))
        self.bdhmm = PermutedDiscreteHMM(self.initial_logits,
                                         self.transition_logits,
                                         self.observation_dist,
                                         self.possible_perms)
        self.shmm = SampleableDiscreteHMM(self.initial_logits,
                                          self.transition_logits,
                                          self.observation_dist)

    def test_something(self):
        max_t = 10
        num_samples = 1000
        x, y = self.shmm.sample((num_samples, max_t))
        ground_truth = x[..., 0]
        while ((ground_truth.unsqueeze(-1) == self.testing_states).sum(-2) == 0).any():
            x, y = self.shmm.sample((num_samples, max_t))
            ground_truth = x[..., 0]
        log_post_dist = self.shmm.posterior_log_initial_state_dist(y)
        ep = bayes_perm_hmm.postprocessing.PostDistEmpiricalPostprocessor(ground_truth, self.testing_states, self.shmm.initial_logits.shape[-1], log_post_dist)
        mr = ep.misclassification_rates()
        print(mr)
        self.assertTrue(torch.allclose(mr.confusions.rate.sum(-1)[self.testing_states], torch.tensor(1.)))
        self.assertTrue((mr.confusions.interval[0] <= mr.confusions.rate).all())
        self.assertTrue((mr.confusions.interval[1] >= mr.confusions.rate).all())
        v = self.bdhmm.get_perms(y)
        b_log_post_dist = v.history.partial_post_log_init_dists[..., -1, :]
        bep = bayes_perm_hmm.postprocessing.PostDistEmpiricalPostprocessor(ground_truth, self.testing_states, self.shmm.initial_logits.shape[-1], b_log_post_dist)
        mr = bep.misclassification_rates()
        print(mr)
        self.assertTrue(torch.allclose(mr.confusions.rate.sum(-1)[self.testing_states], torch.tensor(1.)))
        self.assertTrue((mr.confusions.interval[0] <= mr.confusions.rate).all())
        self.assertTrue((mr.confusions.interval[1] >= mr.confusions.rate).all())

        observation_params = self.observation_dist._param
        bright_state = observation_params.argmax(-1)
        dark_state = observation_params.argmin(-1)
        testing_states = torch.tensor([bright_state, dark_state], dtype=int)
        ic = InterruptedClassifier(
            self.observation_dist,
            testing_states,
        )
        train_x, train_y = self.shmm.sample((num_samples, max_t))
        ground_truth = train_x[..., 0]
        while ((ground_truth.unsqueeze(-1) == self.testing_states).sum(-2) == 0).any():
            train_x, train_y = self.shmm.sample((num_samples, max_t))
            ground_truth = train_x[..., 0]
        _ = train(ic, train_y, train_x[..., 0], self.shmm.initial_logits.shape[-1])
        ic_results = ic.classify(y)
        ip = bayes_perm_hmm.postprocessing.InterruptedEmpiricalPostprocessor(
            ground_truth,
            ic.testing_states,
            self.shmm.initial_logits.shape[-1],
            *ic_results
        )
        mr = ip.misclassification_rates()
        print(mr)
        self.assertTrue(torch.allclose(mr.confusions.rate.sum(-1)[ic.testing_states], torch.tensor(1.)))
        self.assertTrue((mr.confusions.interval[0] <= mr.confusions.rate).all())
        self.assertTrue((mr.confusions.interval[1] >= mr.confusions.rate).all())

        all_data = torch.stack([num_to_data(x, max_t) for x in range(2**max_t)])
        all_naive_post = self.shmm.posterior_log_initial_state_dist(all_data)
        bayes_results = self.bdhmm.get_perms(all_data)
        naive_lp = self.shmm.log_prob(all_data)
        np = bayes_perm_hmm.postprocessing.PostDistExactPostprocessor(
            naive_lp,
            all_naive_post,
            self.initial_logits,
            self.testing_states
        )
        mr = np.misclassification_rates()
        print(mr)
        self.assertTrue(mr.confusions.sum(-1)[self.testing_states].allclose(torch.tensor(1.)))
        bp = bayes_perm_hmm.postprocessing.PostDistExactPostprocessor(
            self.bdhmm.log_prob_with_perm(bayes_results.optimal_perm, all_data),
            bayes_results.history.partial_post_log_init_dists[..., -1, :],
            self.initial_logits,
            self.testing_states,
        )
        mr = bp.misclassification_rates()
        print(mr)
        self.assertTrue(mr.confusions.sum(-1)[self.testing_states].allclose(torch.tensor(1.)))
        _ = exact_train(ic, all_data, naive_lp, all_naive_post, self.initial_logits)
        ic_results = ic.classify(all_data)
        ip = bayes_perm_hmm.postprocessing.InterruptedExactPostprocessor(
            naive_lp,
            all_naive_post,
            self.initial_logits,
            ic.testing_states,
            ic_results,
        )
        mr = ip.misclassification_rates()
        print(mr)
        self.assertTrue(mr.confusions.sum(-1)[ic.testing_states].allclose(torch.tensor(1.)))


if __name__ == '__main__':
    unittest.main()
