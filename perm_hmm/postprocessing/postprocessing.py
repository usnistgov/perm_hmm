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
    floss[~loss] = ZERO
    return floss.log()


class ExactPostprocessor(object):
    """
    An abstract class for postprocessing the data obtained from running
    a simulation to compute the exact misclassification rate of a model.

    .. seealso:: Instances :py:class:`InterruptedExactPostprocessor`,
        :py:class:`PostDistExactPostprocessor`
    """

    def __init__(self, log_joint, testing_states, classifications, score=None):
        """
        :param torch.Tensor log_joint:
        :param testing_states:
        :param classifications:
        :param score:
        """
        if not len(classifications.shape) == 1:
            raise ValueError("Classifications do not have the right shape")
        if not len(log_joint.shape) == 2:
            raise ValueError("log_joint does not have the right shape.")
        if not log_joint.shape[-1] == classifications.shape[-1]:
            raise ValueError("Classifications should have same shape as log_joint.")
        if (score is not None) and (score.shape != classifications):
            raise ValueError("Score should have same shape as classifications.")
        if not log_joint[testing_states].logsumexp(-1).logsumexp(-1).isclose(torch.tensor(0.), atol=5e-7):
            raise ValueError("Prior has weight outside of testing_states")
        if not set(classifications.unique().tolist()).issubset(set(testing_states.tolist())):
            raise ValueError("Not all elements of self.classifications are"
                             " elements of testing_states.")
        if not (testing_states.unique().sort().values == testing_states.sort().values).all():
            raise ValueError("testing_states must be unique")
        self.testing_states = testing_states
        self.log_joint = log_joint
        self.classifications = classifications
        self.score = score

    def log_risk(self, log_loss):
        ll = log_loss(self.testing_states, self.classifications)
        return (self.log_joint[self.testing_states] + ll).logsumexp(-1).logsumexp(-1)

    def log_misclassification_rate(self):
        return self.log_risk(log_zero_one)

    def log_confusion_matrix(self):
        subset_log_joint = self.log_joint[self.testing_states]
        log_prior = subset_log_joint.logsumexp(-1)
        if (log_prior.exp() < 5e-7).any():
            raise UserWarning("Not enough weight on self.testing_states, numerical results will be unreliable")
        log_total = log_prior.logsumexp(-1)
        subset_log_joint -= log_total
        log_data_given_state = subset_log_joint - log_prior.unsqueeze(-1)
        one_hot = self.testing_states.unsqueeze(-1) == self.classifications
        f_one_hot = one_hot.float()
        f_one_hot[~one_hot] = ZERO
        log_one_hot = f_one_hot.log()
        log_confusion_rates = (log_data_given_state.unsqueeze(-2) + \
            log_one_hot.unsqueeze(-3)).logsumexp(-1)
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

        :param torch.Tensor postselect_mask: bool. indicating whether or not to
            keep the data. True corresponds to keep.

            shape ``(n_runs,)``

        :returns: ExactPostprocessor, or subclass thereof. A postselected
            version of self.
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
    An abstract class for postprocessing the data obtained from running a
    simulation to compute the approximate misclassification rate of a model.
    """

    def __init__(self, ground_truth, testing_states, classifications, score=None):
        """
        The minimal things which are needed to produce misclassification rates.
        Requires that classified_bright and classified_dark are specified in a
        subclass init.

        :param torch.Tensor ground_truth: Indicates which runs were generated
        from which initial state.
        """
        if not (testing_states.unique().sort().values == testing_states.sort().values).all():
            raise ValueError("testing_states must be unique")
        total_truths = (ground_truth.unsqueeze(-1) == testing_states).sum(-2)
        if (total_truths == 0).any():
            raise ValueError("There are some states for which there are no runs which realized them. Try a bigger sample size.")
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
        all_pairs = torch.stack(
            tuple(reversed(torch.meshgrid(self.testing_states, self.testing_states)))).T
        ground_truth, classifications = torch.broadcast_tensors(self.ground_truth, self.classifications)
        confusions = (all_pairs == torch.stack(
            (ground_truth, classifications),
            dim=-1,
        ).unsqueeze(-2).unsqueeze(-2)).all(-1).sum(-3)
        total_truths = (ground_truth.unsqueeze(-1) == self.testing_states).sum(-2)
        confusion_rates = confusions / total_truths.unsqueeze(-1).float()
        rstates = torch.arange(len(self.testing_states))
        total_misclassifications = (total_truths - confusions[..., rstates, rstates]).sum(-1)
        avg_misclassification_rate = total_misclassifications.float() / total_truths.sum(-1)
        conf_ints = clopper_pearson(1-confidence_level, confusions, total_truths.unsqueeze(-1))
        avg_conf_int = clopper_pearson(1-confidence_level, total_misclassifications, total_truths.sum(-1))
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
        if (~postselection_mask).all():
            raise ValueError("Can't throw out all the data.")
        if self.score is None:
            p_score = self.score
        else:
            p_score = self.score[postselection_mask]
        return EmpiricalPostprocessor(
            self.ground_truth[postselection_mask],
            self.testing_states,
            self.classifications[postselection_mask],
            p_score,
        )
