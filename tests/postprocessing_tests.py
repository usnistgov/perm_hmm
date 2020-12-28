import unittest
import torch
import torch.distributions
import pyro.distributions as dist
import perm_hmm.models.hmms
from perm_hmm.strategies.min_ent import MinEntropySelector
from perm_hmm.models.hmms import DiscreteHMM, PermutedDiscreteHMM
from perm_hmm.classifiers.interrupted import InterruptedClassifier
from perm_hmm.training.interrupted_training import train_ic, exact_train_ic
import perm_hmm.simulations.map_postprocessors as map
import perm_hmm.simulations.interrupted_postprocessors as inp
from perm_hmm.util import transpositions, num_to_data
from perm_hmm.classifiers.generic_classifiers import MAPClassifier

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
                                         self.observation_dist)
        self.perm_selector = MinEntropySelector(self.possible_perms, self.bdhmm, calibrated=True, save_history=True)
        self.shmm = DiscreteHMM(self.initial_logits,
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
        ep = map.PostDistEmpiricalPostprocessor(ground_truth, self.testing_states, self.shmm.initial_logits.shape[-1], log_post_dist)
        mr = ep.misclassification_rates()
        print(mr)
        self.assertTrue(torch.allclose(mr.confusions.rate.sum(-1)[self.testing_states], torch.tensor(1.)))
        self.assertTrue((mr.confusions.interval[0] <= mr.confusions.rate).all())
        self.assertTrue((mr.confusions.interval[1] >= mr.confusions.rate).all())
        v = self.perm_selector.get_perms(y, -1)
        hist = self.perm_selector.calc_history
        b_log_post_dist = hist[b"dist_array"][..., -1, :]
        bep = map.PostDistEmpiricalPostprocessor(ground_truth, self.testing_states, self.shmm.initial_logits.shape[-1], b_log_post_dist)
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
        while ((ground_truth.unsqueeze(-1) == testing_states).sum(-2) == 0).any():
            train_x, train_y = self.shmm.sample((num_samples, max_t))
            ground_truth = train_x[..., 0]
        _ = train_ic(ic, testing_states, train_y, train_x[..., 0], self.shmm.initial_logits.shape[-1])
        ic_results = ic.classify(y, testing_states)
        ip = inp.InterruptedEmpiricalPostprocessor(
            ground_truth,
            testing_states,
            self.shmm.initial_logits.shape[-1],
            ic_results
        )
        mr = ip.misclassification_rates()
        print(mr)
        self.assertTrue(torch.allclose(mr.confusions.rate.sum(-1)[testing_states], torch.tensor(1.)))
        self.assertTrue((mr.confusions.interval[0] <= mr.confusions.rate).all())
        self.assertTrue((mr.confusions.interval[1] >= mr.confusions.rate).all())

        all_data = torch.stack([num_to_data(x, max_t) for x in range(2**max_t)])
        all_naive_post = self.shmm.posterior_log_initial_state_dist(all_data)
        naive_lp = self.shmm.log_prob(all_data)
        map_class = MAPClassifier(self.shmm)
        classifications = map_class.classify(all_data, self.testing_states)
        np = map.PostDistExactPostprocessor(
            naive_lp,
            all_naive_post,
            self.initial_logits,
            self.testing_states,
            classifications,
        )
        mr = np.misclassification_rates()
        print(mr)
        self.assertTrue(mr.confusions.sum(-1)[self.testing_states].allclose(torch.tensor(1.)))
        self.perm_selector.reset(save_history=True)
        bayes_results = self.perm_selector.get_perms(all_data, -1)
        hist = self.perm_selector.calc_history
        b_map_class = MAPClassifier(self.bdhmm.expand_with_perm(bayes_results))
        b_classifications = b_map_class.classify(all_data, self.testing_states)
        bp = map.PostDistExactPostprocessor(
            self.bdhmm.log_prob_with_perm(all_data, bayes_results),
            hist[b"dist_array"][..., -1, :],
            self.initial_logits,
            self.testing_states,
            b_classifications,
        )
        mr = bp.misclassification_rates()
        print(mr)
        self.assertTrue(mr.confusions.sum(-1)[self.testing_states].allclose(torch.tensor(1.)))
        _ = exact_train_ic(ic, self.testing_states, all_data, naive_lp, all_naive_post, self.initial_logits)
        ic_results = ic.classify(all_data, self.testing_states, verbosity=1)
        ip = inp.InterruptedExactPostprocessor(
            naive_lp,
            all_naive_post,
            self.initial_logits,
            self.testing_states,
            ic_results,
        )
        mr = ip.misclassification_rates()
        print(mr)
        self.assertTrue(mr.confusions.sum(-1)[self.testing_states].allclose(torch.tensor(1.)))

    def test_post_dist_emprirical_singleton(self):
        n = 5
        testing_states = torch.tensor([0, 1])
        hmm = perm_hmm.models.hmms.random_phmm(n)
        self.perm_selector.reset(save_history=True)
        x, y = hmm.sample((100,), perm_selector=self.perm_selector)
        perm = self.perm_selector.perm_history
        plisd = hmm.posterior_log_initial_state_dist(y, perm)
        ep = map.PostDistEmpiricalPostprocessor(x, testing_states, n, plisd)
        ep.misclassification_rates()


if __name__ == '__main__':
    unittest.main()
