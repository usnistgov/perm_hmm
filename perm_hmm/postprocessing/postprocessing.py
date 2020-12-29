"""
Classes to be used for postprocessing data after a simulation.
"""

import numpy as np
import torch
from scipy.stats import beta

import perm_hmm.return_types
from perm_hmm.return_types import RateWithInterval, \
    AllRatesWithIntervals, AllRates
from perm_hmm.util import entropy
import perm_hmm.classifiers.interrupted
from perm_hmm.util import ZERO


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


def log_zero_one(state: torch.Tensor, classification: torch.Tensor):
    loss = classification.unsqueeze(-2) != state.unsqueeze(-1)
    floss = loss.float()


class ExactPostprocessor(object):
    """
    An abstract class for postprocessing the data obtained from running
    a simulation to compute the exact misclassification rate of a model.

    .. seealso:: Instances :py:class:`InterruptedExactPostprocessor`,
        :py:class:`PostDistExactPostprocessor`
    """

    def __init__(self, log_joint, testing_states, classifications=None, score=None):
        """
        :param torch.Tensor log_joint:
        :param testing_states:
        :param classifications:
        :param score:
        """
        self.log_joint = log_joint
        self.testing_states = testing_states
        self.log_data_given_state = \
            self.log_post_dist + self.log_prob.unsqueeze(-1) - \
            self.log_prior_dist.unsqueeze(-2)
        self.classifications = classifications
        self.score = score

    def risk(self, log_loss):
        pass

    def misclassification_rates(self) -> AllRates:
        """
        Computes the misclassification rates.

        :returns: A :py:class:`AllRates`, containing

            .confusions: The tensor the probability of inferring state i given
                the state was j, where the state which is conditioned on is the
                -2 dimension.

            .average: The tensor containing the average misclassification rate,
                averaged over the initial state probability distribution.

        :raises NotImplementedError: if the variables `classified_dark` and
            `classified_bright` are not defined in the instance subclass init.
        """
        if self.classifications is None:
            raise NotImplementedError(
                "Must define the classifications in the subclass init."
            )
        masks = self.testing_states == self.classifications.unsqueeze(-1)
        fmasks = masks.float()
        fmasks[~masks] = ZERO
        bool_not_eye = ~torch.eye(len(self.testing_states), dtype=bool)
        f_not_eye = bool_not_eye.float()
        f_not_eye[~bool_not_eye] = ZERO
        log_confusion_rates = (self.log_data_given_state[:, self.testing_states].unsqueeze(-1) + \
            fmasks.log().unsqueeze(-2)).logsumexp(-3)
        log_average_rate = (self.log_prior_dist[self.testing_states].unsqueeze(-1) + \
            log_confusion_rates + f_not_eye.log()).logsumexp(-2).logsumexp(-1)
        total_num_states = self.log_prior_dist.shape[-1]
        confusion_rates = torch.full(self.classifications.shape[:-1] + (total_num_states, total_num_states), ZERO)
        confusion_rates[(...,) + torch.meshgrid(self.testing_states, self.testing_states)] = log_confusion_rates.exp()
        average_rate = log_average_rate.exp()
        return AllRates(
            confusion_rates,
            average_rate,
        )

    def postselected_misclassification(self, prob_to_keep):
        """
        Given a total probability to keep, gives the misclassification rate of
        the model restricted to the domain containing that probability with the
        best score.

        This method is necessary in spite of the
        :py:meth:`ExactPostprocessor.postselect` method because we cannot
        guarantee that the amount of probability kept after using that method
        is exactly the desired probability.

        :param float prob_to_keep: The probability to keep.

        :returns: :py:class:`AllRates` containing:

            .false_positive_rate: :py:class:`torch.Tensor`, float

                The probability that the model will conclude bright while the
                true initial state was dark when restriced to the desired
                domain.

                shape ``()``

            .false_negative_rate: :py:class:`torch.Tensor`, float

                The probability that the model will conclude dark while the
                true initial state was bright when restriced to the desired
                domain.

                shape ``()``

            .average_rate: :py:class:`torch.Tensor`, float

                The average under the prior initial distribution for the other
                two rates.

                shape ``()``

        :raises ValueError: if you try to throw away all the data.
        """
        prob = self.log_prob.exp()
        enum_prob_score = sorted(
            enumerate(zip(prob, self.score)),
            key=lambda x: x[1][1],
        )
        enum = torch.tensor([tup[0] for tup in enum_prob_score])
        sort_prob = torch.tensor([tup[1][0] for tup in enum_prob_score])
        throw = (sort_prob.cumsum(-1) < 1 - prob_to_keep)
        if not throw.any():
            boundary = -1
        else:
            boundary = throw.nonzero().squeeze().max()
        if boundary == len(prob):
            raise ValueError("Can't throw away all the data.")
        boundary_onehot = torch.zeros(len(prob), dtype=bool)
        boundary_onehot[boundary + 1] = True
        mask = (~(throw | boundary_onehot))[enum.argsort()]
        kept_prob = prob[mask].sum()
        most_rates = self.postselect(mask).misclassification_rates()
        b_mask = boundary_onehot[enum.argsort()]
        b_rates = self.postselect(b_mask).misclassification_rates()
        return AllRates(
            *[
                (kept_prob * r1 + (prob_to_keep - kept_prob) * r2)/prob_to_keep
                for r1, r2 in zip(most_rates, b_rates)
            ]
        )

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

        :param torch.Tensor postselect_mask: bool. indicating whether or not to
            keep the data. True corresponds to keep.

            shape ``(n_runs,)``

        :returns: ExactPostprocessor, or subclass thereof. A postselected
            version of self.
        """
        raise NotImplementedError


class EmpiricalPostprocessor(object):
    """
    An abstract class for postprocessing the data obtained from running a
    simulation to compute the approximate misclassification rate of a model.
    """

    def __init__(self, ground_truth, testing_states, total_num_states, classifications=None, score=None):
        """
        The minimal things which are needed to produce misclassification rates.
        Requires that classified_bright and classified_dark are specified in a
        subclass init.

        :param torch.Tensor ground_truth: Indicates which runs were generated
        from which initial state.
        """
        self.ground_truth = ground_truth
        """
        A :py:class:`torch.Tensor` indicating which runs were generated by which
        initial state. Type int,
        
            shape ``(n_runs,)``
        """
        self.classifications = classifications
        """
        A :py:class:`torch.Tensor`, Type int,
        
            shape ``(n_runs,)``
        """
        self.score = score
        """
        A :py:class:`torch.Tensor` containing the scores of the runs. 
        
        Used for postselection.
        
            shape ``(n_runs,)``
        """
        self.testing_states = testing_states
        """
        A :py:class:`torch.Tensor`, 
        containing the indices of the states to test for.
        """
        self.total_num_states = total_num_states

    def postselection_percentage_mask(self, prob_to_keep):
        if self.score is None:
            raise NotImplementedError(
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

    def misclassification_rates(self, confidence_level=.95):
        """
        Given the actual inference for each run and the ground truth, computes
        the empirical misclassification rates with confidence intervals.

        :param float confidence_level: Indicates the confidence level to
            compute the confidence intervals at.

        :returns: A :py:class:`AllRatesWithIntervals` object, containing
            .false_positive, .false_negative, and .average, which are each
            :py:class:`RateWithInterval` objects containing the
            corresponding misclassification rate along with its confidence
            interval, all at the same level.

        :raises NotImplementedError: if the parameters `classified_bright` or
            `classified_dark` were not specified in the subclass init.
        """
        if self.classifications is None:
            raise NotImplementedError("classified_bright or classified_dark"
                                      "is None. Were these specified in the"
                                      "subclass init?")
        all_pairs = torch.stack(
            tuple(reversed(torch.meshgrid(self.testing_states, self.testing_states)))).T
        ground_truth, classifications = torch.broadcast_tensors(self.ground_truth, self.classifications)
        confusions = (all_pairs == torch.stack(
            (ground_truth, classifications),
            dim=-1,
        ).unsqueeze(-2).unsqueeze(-2)).all(-1).sum(-3)
        total_truths = (self.ground_truth.unsqueeze(-1) == self.testing_states).sum(-2)
        if (total_truths == 0).any():
            raise ValueError("There are some states for which there are no runs which realized them. Try a bigger sample size.")
        confusion_rates = confusions / total_truths.unsqueeze(-1).float()
        rstates = torch.arange(len(self.testing_states))
        total_misclassifications = (total_truths - confusions[..., rstates, rstates]).sum(-1)
        avg_misclassification_rate = total_misclassifications.float() / total_truths.sum(-1)
        conf_ints = clopper_pearson(1-confidence_level, confusions, total_truths.unsqueeze(-1))
        avg_conf_int = clopper_pearson(1-confidence_level, total_misclassifications, total_truths.sum(-1))
        total_num_states = self.total_num_states
        tmp = torch.zeros(self.classifications.shape[:-1] + (total_num_states, total_num_states))
        tmp[(...,) + torch.meshgrid(self.testing_states, self.testing_states)] = confusion_rates
        confusion_rates = tmp
        tmp = torch.zeros((2,) + self.classifications.shape[:-1] + (total_num_states, total_num_states))
        tmp[(...,) + torch.meshgrid(self.testing_states, self.testing_states)] = conf_ints.float()
        conf_ints = tmp
        return AllRatesWithIntervals(
            RateWithInterval(confusion_rates, conf_ints),
            RateWithInterval(avg_misclassification_rate, avg_conf_int),
        )

    def postselect(self, postselection_mask):
        """
        Postselects the data according to the postselection mask.

        :param torch.Tensor postselect_mask: bool.
            A boolean tensor indicating whether or not to
            keep the data. True corresponds to keep.

            shape ``(n_runs,)``

        :returns: EmpiricalPostprocessor, or subclass thereof. A postselected
            version of self.
        """
        raise NotImplementedError
