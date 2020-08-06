"""
Classes to be used for postprocessing data after a simulation.
"""

import numpy as np
import torch
from scipy.stats import beta

import bayes_perm_hmm.return_types
from bayes_perm_hmm.return_types import RateWithInterval, \
    AllRatesWithIntervals, AllRates
from bayes_perm_hmm.util import entropy
import bayes_perm_hmm.interrupted
from bayes_perm_hmm.util import ZERO


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
    """
    An abstract class for postprocessing the data obtained from running
    a simulation to compute the exact misclassification rate of a model.

    .. seealso:: Instances :py:class:`InterruptedExactPostprocessor`,
        :py:class:`PostDistExactPostprocessor`
    """

    def __init__(self, log_prob, log_post_dist, log_prior_dist, testing_states,
                 classifications=None, score=None):
        """
        The minimal things which are needed to produce the misclassification
        rates. Requires that classified_bright and classified_dark are specified
        in a subclass init.

        :param bayes_perm_hmm.return_types.LogProbAndPostDist log_prob_post_dist:
            The log likelihoods and log posterior initial state distributions
            of all the runs. The .log_prob and .log_post_dist should have
            same first dimension.
        :param torch.Tensor log_prior_dist: The log prior initial state distribution.
            shape ``(state_dim,)``
        :param int bright_state: indicates what index corresponds to the bright
            state.
        :param int dark_state: indicates what index corresponds to the dark
            state.
        """
        self.log_prob = log_prob
        """
        The log likelihoods of all the runs.
        
        A :py:class:`torch.Tensor`, float, 
        
            shape ``(n_runs,)``
        """
        self.log_post_dist = log_post_dist
        """
        The log posterior initial state distributions of all the runs to 
        process.
        
        A :py:class:`torch.Tensor`, float, 
        
            shape ``(n_runs,)``
        """
        self.log_prior_dist = log_prior_dist
        """
        The log prior initial state distribution.
        
        A :py:class:`torch.Tensor`, float, 
        
            shape ``(state_dim,)``
        """
        self.testing_states = testing_states
        """
        The states to perform classifications for.
        
        A :py:class:`torch.Tensor`, int,
        
            shape ``(num_testing_states,)``
        """
        self.log_data_given_state = \
            self.log_post_dist + self.log_prob.unsqueeze(-1) - \
            self.log_prior_dist.unsqueeze(-2)
        """
        The log likelihoods of the runs, given the initial states.
        
        A :py:class:`torch.Tensor`, float, 
        
            shape ``(n_runs, state_dim)``
        """
        self.classifications = classifications
        """
        All classifications, :py:class:`torch.Tensor`, int
        
            shape ``(n_runs,)``
        """
        self.score = score
        """
        The score used to postselect the runs.
        
        A :py:class:`torch.Tensor`, float, 
        
            shape ``(n_runs,)``
        """

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


class PostDistExactPostprocessor(ExactPostprocessor):
    """
    Computes statistics from the data obtained from a simulation used to compute
    the exact misclassification rates.
    """

    def __init__(self, log_prob, log_post_dist, log_prior_dist, testing_states,
                 classifications=None, score=None):
        """
        See superclass for details.
        Classifies the runs using the MAP classifier.
        """
        if classifications is None:
            classifications = testing_states[log_post_dist[..., testing_states].argmax(-1)]
        if score is None:
            score = -entropy(log_post_dist)
        super().__init__(log_prob, log_post_dist, log_prior_dist,
                         testing_states, classifications, score)

    def postselect(self, postselect_mask):
        """
        Postselects the data according to a postselection mask.

        See superclass for details on inputs.

        :returns: PostDistExactPostprocessor. Contains a postselected version of the
            data.
        """
        if (~postselect_mask).all():
            raise ValueError("Can't throw out all the data.")
        remaining_log_prob = self.log_prob[postselect_mask].logsumexp(-1)
        postselected_post_dist = self.log_post_dist[postselect_mask]
        postselected_log_prob =\
            self.log_prob[postselect_mask] - remaining_log_prob
        postselected_postprocessor = PostDistExactPostprocessor(
            postselected_log_prob,
            postselected_post_dist,
            self.log_prior_dist,
            self.testing_states,
            self.classifications[postselect_mask],
            self.score[postselect_mask],
        )
        return postselected_postprocessor


class InterruptedExactPostprocessor(ExactPostprocessor):
    """
    Processes data from a numerical experiment which computes the exact
    misclassification rate for the interrupted classifier.
    """

    def __init__(self, log_prob, log_post_dist, log_prior_dist, testing_states,
                 class_break_ratio):
        """
        Initializes the postprocessor.

        :param bayes_perm_hmm.return_types.ClassBreakRatio class_break_ratio:
            Contains:

            :param torch.Tensor classified_bright: bool.
                All classifications
                from the interrupted classifier at all combinations of threshold
                likelihood ratios.

                shape ``(n_runs, num_ratios, num_ratios)``

            :param torch.Tensor break_flag: bool.
                All indications of whether or
                not the classifier terminated the run early.

                shape ``(n_runs, num_ratios, num_ratios)``

            :param torch.Tensor log_like_ratio: float
                All the log likelihood ratios.

                shape ``(n_runs,)``

        See superclass for details on other inputs.
        """
        classifications, break_flag, ratios = class_break_ratio
        self.ratios = ratios
        score = self.ratios.abs()
        super().__init__(
            log_prob,
            log_post_dist,
            log_prior_dist,
            testing_states,
            classifications,
            score,
        )
        self.break_flag = break_flag

    def postselect(self, postselect_mask):
        """
        Postselects the data according to a postselection mask.

        See superclass for details on inputs.

        :returns: InterruptedExactPostprocessor.
            Contains a postselected version of the data.
        """
        if (~postselect_mask).all():
            raise ValueError("Can't throw out all the data.")
        remaining_log_prob = self.log_prob[postselect_mask].logsumexp(-1)
        postselected_post_dist = self.log_post_dist[postselect_mask]
        postselected_log_prob = \
            self.log_prob[postselect_mask] - remaining_log_prob
        postselected_all_classifications = \
            self.classifications[..., postselect_mask]
        postselected_all_break_flag = self.break_flag[..., postselect_mask]
        postselected_all_ratios = self.ratios[..., postselect_mask]
        postselected_postprocessor = InterruptedExactPostprocessor(
            postselected_log_prob,
            postselected_post_dist,
            self.log_prior_dist,
            self.testing_states,
            bayes_perm_hmm.return_types.ClassBreakRatio(
                postselected_all_classifications,
                postselected_all_break_flag,
                postselected_all_ratios,
            ),
        )
        return postselected_postprocessor


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
                "The runs were not scored in the subclass init.")
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


class InterruptedEmpiricalPostprocessor(EmpiricalPostprocessor):
    """
    Processes data from a numerical experiment which computes the empirical
    misclassification rate for the interrupted classifier.
    """

    def __init__(self, ground_truth, testing_states, total_num_states, classifications, break_flag=None, log_like_ratio=None):
        """
        Initializes the postprocessor.

        Due to the varied use cases, we break out each of the elemnts of the
        :py:class:`ClassBreakRatio`. We can also initialize this as::

            InterruptedEmpiricalPostprocessor(
                actually_bright,
                actually_dark,
                *class_break_ratio
            )

        where ``class_break_ratio`` is a :py:class:`ClassBreakRatio`

        :param torch.Tensor actually_bright: bool.
            Ground truth of which runs were actually bright.

            shape ``(n_runs,)``

        :param torch.Tensor actually_dark: bool.
            Ground truth of which runs were actually dark.

            shape ``(n_runs,)``

        :param torch.Tensor classified_bright: bool.
            Indicates whether the run was classified bright.

            shape ``(n_runs,)``

        :param torch.Tensor break_flag: bool.
            Indicates whether the run was
            terminated before the full detection cycle.

            shape ``(n_runs,)``

        :param torch.Tensor log_like_ratio: float.
            The final log likelihood ratio of the runs.

            shape ``(n_runs,)``
        """
        if log_like_ratio is not None:
            score = log_like_ratio.abs()
        else:
            score = None
        super().__init__(ground_truth, testing_states, total_num_states, classifications, score)
        self.break_flag = break_flag
        """Indicates whether the run was interrupted."""
        self.log_like_ratio = log_like_ratio
        """The final log likelihood ratio of the runs."""

    def postselect(self, postselection_mask):
        """
        Postselects the postprocessor.

        See superclass for details.
        """
        if (~postselection_mask).all():
            raise ValueError("Can't throw out all the data.")
        return InterruptedEmpiricalPostprocessor(
            self.ground_truth[..., postselection_mask],
            self.testing_states,
            self.total_num_states,
            self.classifications[..., postselection_mask],
            self.break_flag[..., postselection_mask],
            self.log_like_ratio[..., postselection_mask],
        )


class PostDistEmpiricalPostprocessor(EmpiricalPostprocessor):
    """
    Processes data from a numerical experiment which computes the empirical
    misclassification rate for the HMM classifier.
    """

    def __init__(self, ground_truth, testing_states, total_num_states, log_post_dist):
        """
        Initializes the postprocessor. Uses the maximum a posteriori estimator
        for the initial state inference.

        We use the negative posterior initial state entropy as the score.

        :param torch.Tensor log_post_dist: 
            The posterior log initial state distribution of the
            runs according the the model used to classify.

            shape ``(n_runs, state_dim)``

        See superclass for more details on other parameters.
        """
        self.log_post_dist = log_post_dist
        """
        The posterior log initial state distribution. A :py:class:`torch.Tensor`
        of

            shape ``(n_runs, state_dim)``
        """
        classifications = testing_states[self.log_post_dist[..., testing_states].argmax(-1)]
        score = -entropy(self.log_post_dist)
        super().__init__(ground_truth, testing_states, total_num_states, classifications, score)

    def postselect(self, postselection_mask):
        """
        Postselects the postprocessor.

        See superclass for details.
        """
        if (~postselection_mask).all():
            raise ValueError("Can't throw out all the data.")
        return PostDistEmpiricalPostprocessor(
            self.ground_truth[..., postselection_mask],
            self.testing_states,
            self.total_num_states,
            self.log_post_dist[..., postselection_mask, :],
        )
