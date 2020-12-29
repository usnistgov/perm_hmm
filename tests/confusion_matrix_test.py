import unittest
import torch
import torch.distributions as dist
from perm_hmm.postprocessing.postprocessing import EmpiricalPostprocessor, ExactPostprocessor


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.num_states = 10
        self.testing_states = torch.tensor([0, 3, 4], dtype=int)
        self.num_runs = 1000
        self.classifications = torch.randint(self.num_states, (self.num_runs,))
        self.ground_truth = torch.randint(self.num_states, (self.num_runs,))
        self.empirical_postprocessor = EmpiricalPostprocessor(self.ground_truth, self.testing_states, self.num_states, self.classifications)
        n = self.num_runs*self.num_states
        fake_joint = dist.Dirichlet(torch.full((n,), 1./n)).sample().log()
        fake_joint = fake_joint.reshape((self.num_runs, self.num_states))
        fake_lp = fake_joint.logsumexp(-1)
        fake_prior = fake_joint.logsumexp(-2)
        fake_post_dist = fake_joint - fake_prior
        self.restricted_classifications = self.testing_states[torch.randint(len(self.testing_states), (self.num_runs,))]
        self.exact_postprocessor = ExactPostprocessor(fake_lp, fake_post_dist, fake_prior, self.testing_states, self.restricted_classifications)

    def test_confusion_matrix(self):
        ((all_rates, all_ints), (avg_rate, avg_int)) = self.empirical_postprocessor.misclassification_rates(.95)
        self.assertTrue(torch.all(all_rates <= 1))
        for i in range(len(self.testing_states)):
            for j in range(len(self.testing_states)):
                self.assertTrue(((self.classifications[self.ground_truth == self.testing_states[i]] == self.testing_states[j]).sum()/(self.ground_truth == self.testing_states[i]).sum().float()).isclose(all_rates[self.testing_states[i], self.testing_states[j]]))
        mask = torch.zeros_like(self.ground_truth, dtype=bool)
        for state in self.testing_states:
            mask = mask | (state == self.ground_truth)
        self.assertTrue(((~(self.classifications[mask] == self.ground_truth[mask])).sum() / float(mask.sum())).isclose(avg_rate))
        cov_num_exp = 5000
        cov_truth = torch.randint(self.num_states, (cov_num_exp, self.num_runs))
        cov_classifications = torch.randint(self.num_states, (cov_num_exp, self.num_runs))
        cov_postprocessor = EmpiricalPostprocessor(cov_truth, self.testing_states, self.num_states, cov_classifications)
        results = cov_postprocessor.misclassification_rates(.95)
        avg_coverage = ((results.average.rate > avg_int[0]) &
                        (results.average.rate < avg_int[1])).sum(0) / float(cov_num_exp)
        print(avg_coverage)
        # self.assertTrue(avg_coverage.isclose(torch.tensor(.95), atol=.1))
        all_coverage = ((results.confusions.rate > all_ints[0]) &
                        (results.confusions.rate < all_ints[1])).sum(0) / float(cov_num_exp)
        print(all_coverage)

    def test_exact_post(self):
        confusion_rates, average_rate = self.exact_postprocessor.misclassification_rates()
        self.assertTrue(torch.all(confusion_rates <= 1))
        self.assertTrue((confusion_rates.log() + self.exact_postprocessor.log_prior_dist.unsqueeze(-1))[torch.meshgrid(self.testing_states, self.testing_states)][~torch.eye(len(self.testing_states), dtype=bool)].logsumexp(-1).exp().isclose(average_rate))



if __name__ == '__main__':
    unittest.main()
