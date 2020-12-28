from typing import NamedTuple

import torch


class RateWithInterval(NamedTuple):
    """
    An empirical misclassification rate, along with its confidence interval.

    .. seealso:: :py:class:`AllRatesWithIntervals`, :py:func:`clopper_pearson`
    """
    rate: torch.Tensor
    """:py:class:`torch.Tensor` float. The misclassification rate.
    
        shape arbitrary.
    """
    interval: torch.Tensor
    """:py:class`Interval`. The confidence interval associated."""


class AllRatesWithIntervals(NamedTuple):
    """
    A triple of empirical misclassification rates with confidence intervals.

    .. seealso:: :py:meth:`EmpiricalPostprocessor.misclassification_rates`
    """
    confusions: RateWithInterval
    """:py:class:`RateWithInterval`. 
    The false positive rate and confidence interval."""
    average: RateWithInterval
    """:py:class:`RateWithInterval`. 
    The average misclassification rate and confidence interval."""


class AllRates(NamedTuple):
    """
    Container for exact misclassification rates.

    .. seealso:: :py:meth:`ExactPostprocessor.misclassification_rates`
    """
    confusions: torch.Tensor
    """:py:class:`torch.Tensor` float.
    The exact confusion matrix.
        
        shape arbitrary.
    """
    average: torch.Tensor
    """:py:class:`torch.Tensor` float.
    The exact average misclassification rate.

        shape arbitrary.
    """


hmm_fields = [
    ('states', torch.Tensor),
    ('observations', torch.Tensor),
]

HMMOutput = NamedTuple('HMMOutput', hmm_fields)
perm_hmm_fields = hmm_fields + [('perm', torch.Tensor)]
PermHMMOutput = NamedTuple(
    'PermHMMOutput',
    perm_hmm_fields,
)
PHMMOutHistory = NamedTuple(
    'MinEntHMMOutput',
    perm_hmm_fields + [('history', dict)]
)



class LogProbAndPostDist(NamedTuple):
    """
    Container for the log_prob and posterior distributions.

    .. seealso:: Used as input to
        :py:class:`perm_hmm.postprocessing.ExactPostprocessor`,
        :py:class:`perm_hmm.postprocessing.EmpiricalPostprocessor`
    """
    log_prob: torch.Tensor
    """:py:class:`torch.Tensor`, float.
    The log likelihood of the runs.
    
        shape ``(n_runs,)``
    """
    log_post_dist: torch.Tensor
    """:py:class:`torch.Tensor`, float.
    The log posterior initial state distribution.
    
        shape ``(n_runs, state_dim)``
    """


class PermutedParameters(NamedTuple):
    initial_logits: torch.Tensor
    transition_logits: torch.Tensor
    observation_params: torch.Tensor
    possible_perms: torch.Tensor


class PostYPostS0(NamedTuple):
    r"""
    Contains the posterior output distribution, and the
    posterior initial distribution.

    .. seealso:: return type of :py:meth:`PermutedDiscreteHMM.full_posterior`
    """
    log_post_y: torch.Tensor
    r""":py:class:`torch.Tensor`, float.
    The posterior output distribution :math:`p(y_n | y^{n-1})`

        shape ``(n_outcomes, n_perms)``
    """
    log_post_init: torch.Tensor
    r""":py:class:`torch.Tensor`, float.
    The posterior initial state distribution :math:`p(s_0 | y^{n-1})`

        shape ``(n_outcomes, n_perms, state_dim)``
    """


class GenDistEntropy(NamedTuple):
    """
    Contains the expected posterior entropies and the log posterior
    distributions which generate them.

    .. seealso:: the return type of
        :py:meth:`PermutedDiscreteHMM.expected_entropy`
    """
    log_dists: PostYPostS0
    """:py:class:`PostYPostS0`
    The log distributions used to compute the
    posterior entropy.
    """
    expected_entropy: torch.Tensor
    """:py:class:`torch.Tensor`, float.
    The expected posterior entropy.

        shape ``(n_perms,)``
    """


class PermWithHistory(NamedTuple):
    """
    Another representation of the data returned by
    :py:meth:`PermutedDiscreteHMM.get_perms`.

    .. seealso:: classes :py:class:`PermIndex`, :py:class:`MinEntHistory`,
        :py:class:`PermIndexWithHistory`
    """
    perm: torch.Tensor
    """:py:class:`torch.Tensor`.
    Contains the optimal permtutations applied.
    """
    history: dict

class InterruptedParameters(NamedTuple):
    bright_param: torch.Tensor
    dark_param: torch.Tensor


class ClassBreakRatio(NamedTuple):
    """
    .. seealso:: return type of :py:meth:`InterruptedClassifier.classify`
    """
    classifications: torch.Tensor
    """:py:class:`torch.Tensor`, bool.
    The bright classifications.
        
        shape ``batch_shape``
    """
    break_flag: torch.Tensor
    """:py:class:`torch.Tensor`, bool
    Whether or not the classifier terminated early.
        
        shape ``batch_shape``
    """
    log_like_ratio: torch.Tensor
    """:py:class:`torch.Tensor`, float.
    The final log likelihood ratio.
        
        shape ``batch_shape``
    """


class ExactRun(NamedTuple):
    """
    A data structure for the output of an exact misclassification experiment.

    .. seealso:: The output type of
        :py:meth:`BernoulliSimulator._exact_single_run`
    """
    data: torch.Tensor
    """:py:class:`torch.Tensor`, float.
    Either one run or many.

        shape ``(..., time_dim)``
    """
    bayes_history: PermWithHistory
    """:py:class:`perm_hypo.tests.min_entropy_hmm.PermIndexWithHistory`
    The optimal permutations to have applied, and
    the history of the computation done to generate the optimal permutations.
    """


class HMMOutPostDist(NamedTuple):
    """
    The output of an HMM, along with the log posterior initial state
    distributions which correspond to the runs.

    .. seealso:: Used in :py:class:`EmpiricalRun`
    """
    hmm_output: HMMOutput
    """:py:class:`perm_hmm.sampleable.HMMOutput`
    The data output by an HMM without permutations.
    """
    log_post_dist: torch.Tensor
    """:py:class:`torch.Tensor`, float
    The log posterior initial state distribution.

        shape ``(..., state_dim)``
    """


