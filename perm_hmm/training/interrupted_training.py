import numpy as np
import torch
from perm_hmm.classifiers.interrupted import IIDBinaryIntClassifier, IIDInterruptedClassifier
from perm_hmm.postprocessing import ExactPostprocessor, EmpiricalPostprocessor
# from perm_hmm.loss_functions import binary_zero_one, log_binary_zero_one


def exact_train_ic(ic: IIDInterruptedClassifier, all_data, log_joint, num_ratios=20):
    """
    Train the interrupted classifier using the exact chances of the data occurring.

    The interrupted classifier takes as input a single parameter, the threshold likelihood ratio
    at which to terminate the run and conclude a classification.
    :param bayes_perm_hmm.interrupted.InterruptedClassifier ic: To be trained.
    :param torch.Tensor all_data: all possible runs of data.
    :param torch.Tensor log_joint: Corresponding joint log probability of the data.
        shape (num_states, len(all_data))
    :param num_ratios: number of points to perform the brute force search on.
    :return: A tuple containing the minimum misclassification rate over the searched domain and the corresponding threshold log ratio.
    """
    spaced_ratios = torch.arange(num_ratios, dtype=torch.float)
    misclass_rates = torch.zeros(num_ratios, dtype=torch.float)
    for j in range(num_ratios):
        ic.ratio = spaced_ratios[j]
        interrupted_results = ic.classify(
            all_data,
        )
        iep = ExactPostprocessor(
            log_joint,
            interrupted_results,
        )
        misclass_rates[j] = iep.log_misclassification_rate()
    argmin_rate = torch.tensor(np.argmin(misclass_rates.numpy(), -1))
    min_rate = misclass_rates[argmin_rate]
    ic.ratio = spaced_ratios[argmin_rate]
    # min_rate = misclass_rates.min(-1)
    # ic.ratio = spaced_ratios[min_rate.indices]
    return min_rate


def train_ic(ic: IIDInterruptedClassifier, training_data, ground_truth, num_ratios=20):
    """
    :param bayes_perm_hmm.interrupted.InterruptedClassifier ic: the InterruptedClassifier to train.
    :param training_data: data to train on
    :param ground_truth: The true initial states which generated the data.
    :param num_ratios: The number of points to perform the brute force search on.
    :return: The minimum average misclassification rate
    """
    spaced_ratios = torch.arange(num_ratios, dtype=torch.float)
    misclass_rates = torch.zeros(num_ratios, dtype=torch.float)
    for j in range(num_ratios):
        ic.ratio = spaced_ratios[j]
        interrupted_results = ic.classify(
            training_data,
        )
        iep = EmpiricalPostprocessor(
            ground_truth,
            interrupted_results,
        )
        rates = iep.misclassification_rate()
        misclass_rates[j] = rates[b"rate"]
    argmin_rate = torch.tensor(np.argmin(misclass_rates.numpy(), -1))
    min_rate = misclass_rates[argmin_rate]
    ic.ratio = spaced_ratios[argmin_rate]
    return min_rate


def train_binary_ic(bin_ic: IIDBinaryIntClassifier, training_data, ground_truth, dark_state, bright_state, num_ratios=20):
    """
    Trains the classifier. This is to find the optimal likelihood ratio
    thresholds to minimize classification error.

    :param torch.Tensor training_data: float tensor.
        Data to train the classifier on.

        shape ``(num_samples, time_dim)``

    :param torch.Tensor ground_truth: int tensor.
        Ground truth from an HMM.

        shape ``(num_samples,)``

    :param dark_state: int, Indicates which state is the dark state. Needed to
        interpret ground_truth.

    :param bright_state: int, Indicates which state is the bright state. Needed to
        interpret ground_truth.

    :param int num_ratios: sets the grid size to perform the brute force
        search for the minimal misclassification rate on.
    """
    try:
        num_samples, max_t = training_data.shape
    except ValueError as e:
        raise ValueError(
            "Training data must have shape (num_samples, max_t)") from e
    ratios = torch.arange(num_ratios, dtype=torch.float)
    rates = torch.empty((num_ratios, num_ratios), dtype=torch.float)
    for i in range(len(ratios)):
        for j in range(len(ratios)):
            bin_ic.bright_ratio = ratios[i]
            bin_ic.dark_ratio = ratios[j]
            interrupted_results = bin_ic.classify(training_data, verbosity=0).int()
            iep = EmpiricalPostprocessor(
                ground_truth,
                interrupted_results,
            )
            # rate = iep.risk(binary_zero_one(dark_state, bright_state))
            rate = iep.misclassification_rate()[b"rate"]
            rates[i, j] = rate
    # ind = divmod(np.argmin(rates.numpy()), rates.shape[1])
    ind = np.unravel_index(np.argmin(rates.numpy()), rates.shape)
    bin_ic.bright_ratio = ratios[ind[0]]
    bin_ic.dark_ratio = ratios[ind[1]]

def exact_train_binary_ic(bin_ic: IIDBinaryIntClassifier, all_data, log_joint, num_ratios=20):
    """
    Trains the classifier. This is to find the optimal likelihood ratio
    thresholds to minimize classification error.

    :param torch.Tensor all_data: All possible runs of data.

        shape ``(num_runs, time_steps)``

    :param torch.Tensor log_joint: Corresponding log joint likelihoods.

        shape ``(num_states, num_runs)``

    :param dark_state: int, Indicates which state is the dark state. Needed to
        interpret ground_truth.

    :param bright_state: int, Indicates which state is the bright state. Needed to
        interpret ground_truth.

    :param int num_ratios: sets the grid size to perform the brute force
        search for the minimal misclassification rate on.
    """
    ratios = torch.arange(num_ratios, dtype=torch.float)
    rates = torch.empty((num_ratios, num_ratios), dtype=torch.float)
    for i in range(len(ratios)):
        for j in range(len(ratios)):
            bin_ic.bright_ratio = ratios[i]
            bin_ic.dark_ratio = ratios[j]
            interrupted_results = bin_ic.classify(
                all_data,
            ).int()
            iep = ExactPostprocessor(
                log_joint,
                interrupted_results,
            )
            # rates[i, j] = iep.log_risk(log_binary_zero_one(dark_state, bright_state))
            rates[i, j] = iep.log_misclassification_rate()
    # ind = divmod(rates.argmin().item(), rates.shape[1])
    ind = np.unravel_index(np.argmin(rates.numpy()), rates.shape)
    bin_ic.bright_ratio = ratios[ind[0]]
    bin_ic.dark_ratio = ratios[ind[1]]
