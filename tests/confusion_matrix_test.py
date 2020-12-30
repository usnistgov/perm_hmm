import unittest
import torch
import torch.distributions as dist
from perm_hmm.postprocessing import EmpiricalPostprocessor, ExactPostprocessor
from perm_hmm.util import ZERO


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.num_states = 10
        self.testing_states = torch.tensor([0, 3, 4], dtype=int)
        self.num_runs = 1000
        self.classifications = self.testing_states[torch.randint(len(self.testing_states), (self.num_runs,))]
        self.ground_truth = self.testing_states[torch.randint(len(self.testing_states), (self.num_runs,))]
        self.empirical_postprocessor = EmpiricalPostprocessor(self.ground_truth, self.classifications)
        n = self.num_runs*self.num_states
        fake_joint = dist.Dirichlet(torch.full((n,), 1./n)).sample().log()
        fake_joint = fake_joint.reshape((self.num_states, self.num_runs))
        log_zero = torch.tensor(ZERO).log()
        fake_joint[fake_joint < log_zero] = log_zero
        not_states = torch.tensor(list(set(range(self.num_states)).difference(set(self.testing_states.tolist()))))
        fake_joint[not_states] = torch.tensor(ZERO).log()
        fake_joint -= fake_joint.logsumexp(-1).logsumexp(-1)
        self.initial_logits = fake_joint.logsumexp(-1)
        while (self.initial_logits[self.testing_states].exp() < 1e-6).any():
            fake_joint = dist.Dirichlet(torch.full((n,), 1. / n)).sample().log()
            fake_joint = fake_joint.reshape((self.num_states, self.num_runs))
            log_zero = torch.tensor(ZERO).log()
            fake_joint[fake_joint < log_zero] = log_zero
            not_states = torch.tensor(list(
                set(range(self.num_states)).difference(
                    set(self.testing_states.tolist()))))
            fake_joint[not_states] = torch.tensor(ZERO).log()
            fake_joint -= fake_joint.logsumexp(-1).logsumexp(-1)
            self.initial_logits = fake_joint.logsumexp(-1)
        self.restricted_classifications = self.testing_states[torch.randint(len(self.testing_states), (self.num_runs,))]
        self.exact_postprocessor = ExactPostprocessor(fake_joint, self.restricted_classifications)

    def test_confusion_matrix(self):
        rate_dict = self.empirical_postprocessor.misclassification_rate(.95)
        avg_rate = rate_dict[b"rate"]
        avg_int = torch.tensor([rate_dict[b"lower"], rate_dict[b"upper"]])
        conf_dict = self.empirical_postprocessor.confusion_matrix(.95)
        all_rates = conf_dict[b'rate']
        all_ints = torch.stack([conf_dict[b'lower'], conf_dict[b'upper']])
        self.assertTrue(torch.all(all_rates[~torch.isnan(all_rates)] <= 1))
        for i in range(len(self.testing_states)):
            for j in range(len(self.testing_states)):
                total_i = (self.ground_truth == self.testing_states[i]).sum().float()
                total_ij = (self.classifications[self.ground_truth == self.testing_states[i]] == self.testing_states[j]).sum()
                frequency = total_ij/total_i
                self.assertTrue((frequency).isclose(all_rates[self.testing_states[i], self.testing_states[j]]))
        mask = torch.zeros_like(self.ground_truth, dtype=bool)
        for state in self.testing_states:
            mask = mask | (state == self.ground_truth)
        self.assertTrue(((~(self.classifications[mask] == self.ground_truth[mask])).sum() / float(mask.sum())).isclose(avg_rate))
        # cov_num_exp = 5000
        # cov_truth = torch.randint(self.num_states, (cov_num_exp, self.num_runs))
        # cov_classifications = torch.randint(self.num_states, (cov_num_exp, self.num_runs))
        # cov_postprocessor = EmpiricalPostprocessor(cov_truth, cov_classifications)
        # results = cov_postprocessor.misclassification_rate(.95)
        # avg_coverage = ((results[b"rate"] > avg_int[0]) &
        #                 (results[b"rate"] < avg_int[1])).sum(0) / float(cov_num_exp)
        # print(avg_coverage)
        # all_coverage = ((results[b"rate"] > all_ints[0]) &
        #                 (results[b"rate"] < all_ints[1])).sum(0) / float(cov_num_exp)
        # print(all_coverage)

    def test_exact_post(self):
        log_average_rate = self.exact_postprocessor.log_misclassification_rate()
        log_confusion_matrix = self.exact_postprocessor.log_confusion_matrix()
        confusion_rates = log_confusion_matrix.exp()
        average_rate = log_average_rate.exp()
        self.assertTrue(torch.all(confusion_rates[~torch.isnan(confusion_rates)] <= 1))
        log_prior = self.exact_postprocessor.log_joint.logsumexp(-1)
        valid_matrix = log_confusion_matrix[torch.meshgrid(self.testing_states, self.testing_states)]
        valid_prior = log_prior[self.testing_states].unsqueeze(-1)
        test_log_rate = (valid_matrix + valid_prior)[~torch.eye(len(self.testing_states), dtype=bool)].logsumexp(-1)
        test_rate = test_log_rate.exp()
        self.assertTrue(test_rate.isclose(average_rate, atol=1e-5))



if __name__ == '__main__':
    unittest.main()
