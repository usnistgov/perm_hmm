import torch
import perm_hmm.classifiers.interrupted
import perm_hmm.simulations.postprocessing


def exact_train_ic(ic, testing_states, all_data, log_probs, log_post_dist, log_prior_dist,
                   num_ratios=20):
    """
    Train the interrupted classifier using the exact chances of the data occurring.

    The interrupted classifier takes as input a single parameter, the threshold likelihood ratio
    at which to terminate the run and conclude a classification.
    :param bayes_perm_hmm.interrupted.InterruptedClassifier ic: To be trained.
    :param torch.Tensor all_data: all possible runs of data.
    :param torch.Tensor log_probs: Corresponding log likelihoods of the data.
    :param torch.Tensor log_post_dist: Posterior log initial state distributions for all the runs. Last dimension is the state label.
    :param log_prior_dist: True log initial state distribution.
    :param num_ratios: number of points to perform the brute force search on.
    :return: A tuple containing the minimum misclassification rate over the searched domain and the corresponding threshold log ratio.
    """
    spaced_ratios = torch.arange(num_ratios, dtype=torch.float)
    misclass_rates = torch.zeros(num_ratios, dtype=torch.float)
    for j in range(num_ratios):
        ic.ratio = spaced_ratios[j]
        interrupted_results = ic.classify(
            all_data,
            testing_states,
        )
        iep = perm_hmm.simulations.postprocessing.InterruptedExactPostprocessor(
            log_probs,
            log_post_dist,
            log_prior_dist,
            testing_states,
            interrupted_results,
        )
        rates = iep.misclassification_rates()
        misclass_rates[j] = rates.average
    min_rate = misclass_rates.average.min(-1)
    ic.ratio = spaced_ratios[min_rate.indices]
    return min_rate.values


def train_ic(ic, testing_states, training_data, ground_truth, total_num_states, num_ratios=20):
    """
    :param bayes_perm_hmm.interrupted.InterruptedClassifier ic: the InterruptedClassifier to train.
    :param training_data: data to train on
    :param ground_truth: The true initial states which generated the data.
    :param num_ratios: The number of points to perform the brute force search on.
    :return: A tuple containing the minimum average misclassification rate and its corresponding threshold log ratio.
    """
    spaced_ratios = torch.arange(num_ratios, dtype=torch.float)
    misclass_rates = torch.zeros(num_ratios, dtype=torch.float)
    interrupted_results = ic.classify(training_data, spaced_ratios)
    for j in range(num_ratios):
        ic.ratio = spaced_ratios[j]
        interrupted_results = ic.classify(
            training_data,
            testing_states,
        )
        iep = perm_hmm.simulations.postprocessing.InterruptedEmpiricalPostprocessor(
            ground_truth,
            testing_states,
            total_num_states,
            *interrupted_results,
        )
        rates = iep.misclassification_rates()
        misclass_rates[j] = rates.average
    min_rate = misclass_rates.average.min(-1)
    ic.ratio = spaced_ratios[min_rate.indices]
    return min_rate.values
