"""
Classes to be used for postprocessing data after a simulation.
"""
import warnings

import numpy as np
import torch
from scipy.stats import beta

from perm_hmm.util import ZERO
from perm_hmm.loss_functions import zero_one, log_zero_one


def clopper_pearson(alpha, num_successes, total_trials):
    """
    Computes the `exact binomial`_ confidence interval for confidence level
    1-`alpha`.


    This method uses the scipy.stats.beta.ppf function because I couldn't
    find it in the torch framework.

    :param float alpha: between 0 and 1. 1-alpha is the confidence level.

    :param torch.Tensor num_successes: number of "positive" inferences.

        shape arbitrary, but must match that of `total_trials`.

    :param torch.Tensor total_trials: number of total inferences.

        shape arbitrary, but must match that of `num_successes`.

    :returns: A :py:class:`Interval` object, with attributes

        .lower: A :py:class:`torch.Tensor`, float, shape same as num_successes.

        .upper: A :py:class:`torch.Tensor`, float, shape same as num_successes.

    :raises ValueError: if the input tensors are not of the same shape.

    .. _exact binomial: https://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Clopper%E2%80%93Pearson_interval
    """
    if (num_successes > total_trials).any():
        raise ValueError("The number of successful trials cannot be more than"
                         " the total number of trials")
    npalpha = np.array(alpha)
    n_successes = np.array(num_successes)
    trials = np.array(total_trials)
    if (trials == 0).any():
        raise ValueError("The total number of trials should be nonzero")
    n_successes, trials = np.broadcast_arrays(n_successes, trials)
    lower = np.array(beta.ppf(npalpha / 2, n_successes, trials - n_successes + 1))
    upper = np.array(beta.ppf(1 - npalpha / 2, n_successes + 1, trials - n_successes))
    lower[n_successes == 0] = 0
    upper[n_successes == 0] = 1 - (npalpha / 2)**(1 / trials[n_successes == 0])
    lower[n_successes == trials] = (npalpha / 2)**(1 / trials[n_successes == trials])
    upper[n_successes == trials] = 1
    return torch.stack((torch.from_numpy(lower), torch.from_numpy(upper)))


class ExactPostprocessor(object):
    r"""
    A class for postprocessing the data obtained from running
    a simulation to compute the exact misclassification rate of a model.

    Instances of this class have the following attributes:

    ``log_joint``:
        :math:`\log(\mathbb{P}(s_0, y^n))`, represented by a
        :py:class:`~torch.Tensor`, of shape
        ``(num_states, num_possible_outcomes)``

    ``classifications``:
        The inferred initial states, as a :py:class:`~torch.Tensor` of shape
        ``(num_possible_outcomes,)``.

    ``score``:
        The "degree of belief" with which the classification is made,
        as a :py:class:`~torch.Tensor` of shape ``(num_possible_outcomes,)``.
        Can be ``None``.
        This is used for postselection. The runs with the highest score are
        kept.
    """

    def __init__(self, log_joint, classifications, score=None, testing_states=None):
        if not len(classifications.shape) == 1:
            raise ValueError("Classifications must have exactly 1 dimension")
        if not len(log_joint.shape) == 2:
            raise ValueError("log_joint must have exactly two dimensions.")
        if not log_joint.shape[-1] == classifications.shape[-1]:
            raise ValueError("Classifications should have same last dimension as log_joint.")
        if (score is not None) and (score.shape != classifications.shape):
            raise ValueError("Score should have same shape as classifications.")
        self.log_joint = log_joint
        self.classifications = classifications
        self.score = score

    def log_risk(self, log_loss):
        """
        Computes log bayes risk for a given log loss function.

        :param log_loss: log loss function, should take arguments state, classification
            and return the log loss for that pair.
        :return: The log bayes risk for the given function.
        """
        states = torch.arange(self.log_joint.shape[-2])
        ll = log_loss(states.unsqueeze(-1), self.classifications.unsqueeze(-2))
        return (self.log_joint + ll).logsumexp(-1).logsumexp(-1)

    def log_misclassification_rate(self):
        return self.log_risk(log_zero_one)

    def log_confusion_matrix(self):
        r"""
        Computes the elementwise log of the confusion matrix,
        :math:`\mathbb{P}(\hat{s}|s)`
        """
        log_prior = self.log_joint.logsumexp(-1)
        nonzero_prior = log_prior > torch.tensor(1e-6).log()
        if not nonzero_prior.all():
            warnings.warn("Not all states have nonzero prior, there will be "
                          "NaNs in the confusion matrix.")
        possible_class = torch.arange(self.classifications.max()+1)
        log_data_given_state = self.log_joint - log_prior.unsqueeze(-1)
        one_hot = possible_class.unsqueeze(-1) == self.classifications
        f_one_hot = one_hot.float()
        f_one_hot[~one_hot] = ZERO
        log_one_hot = f_one_hot.log()
        # log_one_hot[~one_hot] = 2*log_one_hot[~one_hot]
        log_confusion_rates = (log_data_given_state.unsqueeze(-2) +
            log_one_hot.unsqueeze(-3)).logsumexp(-1)
        if log_confusion_rates.dtype == torch.double:
            log_confusion_rates[~nonzero_prior] = torch.tensor(float('NaN')).double()
        else:
            log_confusion_rates[~nonzero_prior] = torch.tensor(float('NaN'))
        return log_confusion_rates

    def postselected_misclassification_rate(self, log_prob_to_keep):
        """
        Given a total probability to keep, gives the misclassification rate of
        the model restricted to the domain containing that probability with the
        best score.

        This method is necessary in spite of the
        :py:meth:`ExactPostprocessor.postselect` method because we cannot
        guarantee that the amount of probability kept after using that method
        is exactly the desired probability.

        :param float log_prob_to_keep: The probability to keep.

        :returns: The misclassification rate keeping only the best prob_to_keep of data.

        :raises ValueError: if you try to throw away all the data.
        """
        log_prob = self.log_joint.logsumexp(-1)
        enum_prob_score = sorted(
            enumerate(zip(log_prob, self.score)),
            key=lambda x: x[1][1],
        )
        enum = torch.tensor([tup[0] for tup in enum_prob_score])
        sort_log_prob = torch.tensor([tup[1][0] for tup in enum_prob_score])
        throw = (sort_log_prob.logsumexp(-1) < (1 - log_prob_to_keep.exp()).log())
        if not throw.any():
            boundary = -1
        else:
            boundary = throw.nonzero().squeeze().max()
        if boundary == len(log_prob):
            raise ValueError("Can't throw away all the data.")
        boundary_onehot = torch.zeros(len(log_prob), dtype=bool)
        boundary_onehot[boundary + 1] = True
        mask = (~(throw | boundary_onehot))[enum.argsort()]
        kept_log_prob = log_prob[mask].logsumexp(-1)
        log_most_rate = self.postselect(mask).log_misclassification_rate()
        b_mask = boundary_onehot[enum.argsort()]
        log_b_rate = self.postselect(b_mask).log_misclassification_rate()
        return torch.from_numpy(np.logaddexp((kept_log_prob + log_most_rate).numpy(), (log_prob_to_keep.exp()-kept_log_prob.exp()).log() + log_b_rate)) - log_prob_to_keep

    def postselection_mask(self, threshold_score):
        """
        Returns a mask where the score of the runs is larger than specified.

        :param float threshold_score: The score below which we would like to
            throw out runs.

        :returns: :py:class:`torch.Tensor`, bool. True means keep.

            shape ``(n_runs,)``
        """
        return self.score > threshold_score

    def postselect(self, postselect_mask):
        """
        Postselects the data according to the postselect mask.

        :param torch.Tensor postselect_mask: bool. indicating whether to
            keep the data. True corresponds to keep.

            shape ``(n_runs,)``

        :returns: ExactPostprocessor. A postselected version of self.
        """
        if (~postselect_mask).all():
            raise ValueError("Can't throw out all the data.")
        p_log_joint = self.log_joint[:, postselect_mask]
        p_log_joint -= p_log_joint.logsumexp(-1).logsumexp(-1)
        if self.score is not None:
            p_score = self.score[postselect_mask]
        else:
            p_score = None
        p_classifications = self.classifications[postselect_mask]
        return ExactPostprocessor(p_log_joint, p_classifications, p_score)


class EmpiricalPostprocessor(object):
    """
    A class for postprocessing the data obtained from running a
    simulation to compute the approximate misclassification rate of a model.

    Instances of this class have the following attributes:

    ``ground_truth``:
        The true initial states, shape ``(num_samples,)``

    ``classifications``:
        The inferred initial states, shape ``(num_samples,)``.

    ``score``:
        The "degree of belief" of the classification. shape ``(num_samples,)``.
        Can be ``None``.
        This is used for postselection. The runs with the highest score are
        kept.
    """

    def __init__(self, ground_truth, classifications, score=None):
        """
        The minimal things which are needed to produce misclassification rates.
        Requires that classified_bright and classified_dark are specified in a
        subclass init.

        :param torch.Tensor ground_truth: Indicates which runs were generated
            from which initial state.
        """
        self.ground_truth = ground_truth
        self.classifications = classifications
        self.score = score

    def postselection_percentage_mask(self, prob_to_keep):
        """
        Produces a mask which indicates the top prob_to_keep of data according to
        self.score

        :param prob_to_keep: A float between 0 and 1.
        :return: A boolean tensor indicating which runs to keep

            shape ``(num_runs,)``

        :raises AttributeError: If self.score is None
        """
        if self.score is None:
            raise AttributeError(
                "The data is not scored for postselection.")
        sort_score = self.score.sort()
        mask = torch.zeros_like(self.score, dtype=bool)
        mask[sort_score.indices[:round(len(self.score)*prob_to_keep)]] = True
        return mask

    def postselection_mask(self, threshold_score):
        """
        Masks the runs whose score is too low.

        :param float threshold_score:
            The score below which we will throw out the data.

        :returns: :py:class:`torch.Tensor`, bool.
            True if we want to keep the run.

            shape ``(n_runs,)``
        """
        return self.score > threshold_score

    def risk(self, loss):
        """
        Computes bayes risk by Monte Carlo integration of a loss function.

        :param loss: The loss function to compute bayes risk for.
            Should take a state and a classification and return a float tensor.
        :return: Float tensor, the Bayes risk.
        """
        return loss(self.ground_truth, self.classifications).sum(-1) / torch.tensor(self.classifications.shape[-1]).float()

    def misclassification_rate(self, confidence_level=.95, loss=None):
        """
        Computes misclassification rate and a confidence interval.

        Computed using the Clopper-Pearson binomial interval.

        :param confidence_level: The confidence level to compute a confidence
            interval for.
        :param loss: The loss function to compute the number of misses with.
            defaults to the zero-one loss,

                >>> def loss(ground_truth, classification):
                >>>     return (ground_truth != classification).sum(-1)

        :return: A dict with keys

            b"rate": The misclassification rate as a float tensor.

            b"lower": The lower bound of the confidence interval as a float tensor.

            b"upper": The upper bound of the confidence interval as a float tensor.
        """
        if loss is None:
            def loss(ground_truth, classification):
                return (ground_truth != classification).sum(-1)
        rate = self.risk(zero_one)
        total = torch.tensor(self.classifications.shape[-1])
        num_misses = loss(self.ground_truth, self.classifications)
        interval = clopper_pearson(1-confidence_level, num_misses, total)
        return {b"rate": rate, b"lower": interval[0], b"upper": interval[1]}

    def confusion_matrix(self, confidence_level=.95):
        """
        Computes a confusion matrix and a confidence set for it, at the desired
        confidence level. Requires an R `package`_ "MultinomialCI" for computing
        the confidence set.

        :param confidence_level:
        :return: A dict with keys

            b"matrix": The confusion matrix as a float tensor.

            b"lower": The lower corner of the confidence set as a float tensor.

            b"upper": The upper corner of the confidence set as a float tensor.

        .. _package: https://cran.r-project.org/web/packages/MultinomialCI/MultinomialCI.pdf
        """
        from rpy2.robjects.packages import importr
        from rpy2.robjects import FloatVector
        multinomial_ci = importr("MultinomialCI")
        range_truth = self.ground_truth.max() + 1
        range_class = self.classifications.max() + 1
        possible_class = torch.arange(range_class)
        lower = torch.empty((range_truth, range_class))
        upper = torch.empty((range_truth, range_class))
        matrix = torch.empty((range_truth, range_class))
        for i in range(range_truth.item()):
            mask = self.ground_truth == i
            if mask.any():
                classi = self.classifications[mask]
                counts = (classi == possible_class.unsqueeze(-1)).sum(-1)
                frequencies = counts / mask.sum(-1).float()
                vec = FloatVector(counts)
                ci = multinomial_ci.multinomialCI(vec, 1-confidence_level)
                ci = torch.from_numpy(np.array(ci))
                lower[i] = ci[:, 0]
                upper[i] = ci[:, 1]
                matrix[i] = frequencies
            else:
                warnings.warn("No instances of state {} in ground truth, there"
                              "will be NaNs in confusion matrix.".format(i))
                lower[i] = torch.tensor(float('NaN'))
                upper[i] = torch.tensor(float('NaN'))
                matrix[i] = torch.tensor(float('NaN'))
        return {b"matrix": matrix, b"lower": lower, b"upper": upper}

    def postselect(self, postselection_mask):
        """
        Postselects the data according to the postselection mask.

        :param torch.Tensor postselection_mask: bool.
            A boolean tensor indicating whether to
            keep the data. True corresponds to keep.

            shape ``(n_runs,)``

        :returns: EmpiricalPostprocessor.. A postselected
            version of self.
        """
        if (~postselection_mask).all():
            raise ValueError("Can't throw out all the data.")
        if self.score is None:
            p_score = self.score
        else:
            p_score = self.score[postselection_mask]
        return EmpiricalPostprocessor(
            self.ground_truth[postselection_mask],
            self.classifications[postselection_mask],
            p_score,
        )
