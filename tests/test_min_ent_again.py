from functools import wraps
from functools import reduce
from operator import mul

import numpy as np
import pytest
import torch
import pyro.distributions as dist
from pyro.distributions.hmm import _logmatmulexp

from perm_hmm.models.hmms import PermutedDiscreteHMM
from typing import NamedTuple
from perm_hmm.util import all_strings, id_and_transpositions, ZERO, wrap_index
from example_systems.three_states import three_state_hmm
from tests.min_ent import MinEntropyPolicy as MES


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


class BayesDistribution(object):
    """
    An abstract class for the distributions used to compute the posterior
    expected entropy.

    .. seealso:: Implementations :py:class:`BayesCurrentDistribution`,
        :py:class:`BayesCurrentCondInitialDistribution`,
        :py:class:`BayesInitialDistribution`
    """
    def __init__(self, log_distribution):
        self.logits = log_distribution
        """The log likelihoods for this distribution."""

    def posterior(self, observation_logits):
        """
        The posterior method call signatures are different for the different
        instance distributions. At a minimum, each distribution requires the
        observation logits.

        :param torch.Tensor observation_logits: float.
            The log observations distribution.
        """
        raise NotImplementedError


class BayesInitialDistribution(BayesDistribution):
    r"""
    Stores the prior log initial state distribution.

    This data structure stores :math:`\log(p(s_0 | y^{i-1}))`,
    where :math:`y^{i-1}` is all the data that has been seen so far.

    :param torch.Tensor logits: shape `` batch_shape + (state_dim,)``

    .. seealso:: Instantiated in :py:class:`PermutedDiscreteHMM`
    """

    def posterior(self, observation_logits, prior_s_cond_init):
        r"""
        Given a set of logits for the newly observed data and the distribution
        of the previous state conditional on the initial state, computes the
        posterior initial state distribution.

        This is performed using Bayes' rule, as in the following expressions.

        .. math::
            p(s_0|y^i) &=
            \frac{p(y_i|s_0, y^{i-1}) p(s_0|y^{i-1})}{
            \sum_{s_0} p(y_i|s_0, y^{i-1}) p(s_0|y^{i-1})
            } \\
            p(y_i|s_0, y^{i-1}) &=
            \sum_{s_i} p(y_i|s_i) p(s_i|s_0, y^{i-1}).

        :param torch.Tensor observation_logits:
            shape ``batch_shape + (1, 1, state_dim)``

        :param torch.Tensor prior_s_cond_init:
            shape ``batch_shape + (n_perms, state_dim, state_dim)``,
             where the last dimension is for s_{i+1}

        :returns: either a single posterior distribution or a
            whole posterior distribution tensor

            shape ``batch_shape + (n_perms, state_dim)``
        """
        post_init = (observation_logits + prior_s_cond_init).logsumexp(axis=-1)
        post_init = post_init + self.logits.unsqueeze(-2)
        post_init = post_init - post_init.logsumexp(-1, keepdims=True)
        return post_init


class BayesCurrentDistribution(BayesDistribution):
    r"""
    Denoting the data seen thus far as :math:`y^{i-1}`,
    this class stores the distribution :math:`\log(p(s_i|y^{i-1}))`,
    for all possible permutations to be applied to :math:`s_{i-1}`.

    :param torch.Tensor logits: shape ``batch_shape + (num_perms, state_dim)``

    .. seealso:: Instantiated in :py:class:`PermutedDiscreteHMM`
    """

    def posterior(self, observation_logits, transition_logits,
                  previous_perm_index: torch.Tensor):
        r"""
        Computes the posterior current state distribution, according to
        Bayes rule.

        Denoting :math:`p(y|s) = b(y|s)` as the output distribution and
        :math:`p(s_j | s_i) = a_{ij}` as
        the transition matrix, the Bayes rule update is given by

        .. math::
            p(s_{i+1} | y^i) &= \sum_{s_{i}} a_{\sigma_i(y^i, s_{i})s_{i+1}}
                p(s_i | y^i) \\
            p(s_i | y^i) &= \frac{b(y_i | s_i) p(s_i | y^{i-1})}{
                \sum_{s_i}b(y_i | s_i)p(s_i | y^{i-1})}

        where we have :math:`p(s_i|y^{i-1})` already, and the permutation
        :math:`\sigma_i(y^i, s_i)` is yet to be determined, so
        we compute for all possibilities.

        :param torch.Tensor transition_logits: float. Log transition matrices.

            shape ``batch_shape + (num_perms, state_dim, state_dim)``

        :param torch.Tensor observation_logits: float. The output distributions.

            shape ``batch_shape + (1, 1, state_dim)``

        :param torch.Tensor previous_perm_index: The previously applied
            permutation index.

            shape ``batch_shape``

        :returns: tensor shape ``batch_shape + (num_perms, state_dim)``
        """
        ind = wrap_index(previous_perm_index)
        post_s = self.logits[ind]
        post_s = post_s.unsqueeze(-2).unsqueeze(-2)
        post_s = post_s + observation_logits
        post_s = post_s - post_s.logsumexp(-1, keepdims=True)
        post_s = _logmatmulexp(post_s.float(), transition_logits.float())
        return post_s.squeeze(-2)

    def posterior_y(self, observation_logits):
        r"""
        Computes :math:`p(y_i|y^{i-1})` for all possible permutations to be
        applied to :math:`s_{i-1}`.

        From the equation

            .. math::
                p(y_i|y^{i-1}) = \sum_{s_i} b(y_i|s_i) p(s_i|y^{i-1})

        :param observation_logits:
            shape ``batch_shape + (n_outcomes, 1, 1, state_dim)``
        :returns: shape ``batch_shape + (n_outcomes, n_perms)``
        """
        return (self.logits.unsqueeze(-2) + observation_logits).logsumexp(-1).squeeze(-1)


class BayesCurrentCondInitialDistribution(BayesDistribution):
    r"""
    Stores :math:`p(s_i | s_0, y^{i-1})`,
    for all possible permutations that could be applied to :math:`s_{i-1}`.

    :param torch.Tensor logits:
        shape ``batch_shape + (n_perms, state_dim, state_dim)``,
        the last dimension is for :math:`s_i`,
        and the second to last dimension is for :math:`s_0`.

    .. seealso:: Instantiated in :py:class:`PermutedDiscreteHMM`
    """

    def posterior(self, observation_logits, transition_logits,
                  previous_perm_index: torch.Tensor):
        r"""
        Computes the posterior for all possible permutations.
        Denoting :math:`p(y|s) = b(y|s)` as the output distribution and
        :math:`p(s_j | s_i) = a_{ij}` as
        the transition matrix, the Bayes rule update is given by

        .. math::
            p(s_{i+1} | s_0, y^i) &= \sum_{s_i} a_{\sigma_i(y^i, s_i), s_{i+1}}
                p(s_i | s_0, y^i) \\
            p(s_i| s_0, y^i) &= \frac{b(y_i|s_i) p(s_i | s_0 y^{i-1})}{
                \sum_{s_i} b(y_i|s_i) p(s_i|s_0, y^{i-1})}

        where we have :math:`p(s_i|s_0, y^{i-1})` already, and the permutation
        :math:`\sigma_i(y^i, s_i)` is yet to be determined, so
        we compute for all possibilities.

        :param torch.Tensor transition_logits: float.

            shape ``batch_shape + (num_perms, state_dim, state_dim)``

        :param torch.Tensor observation_logits: float.

            shape ``batch_shape + (1, 1, state_dim)``

        :param torch.Tensor previous_perm_index: int.
            The index which encodes the previous permutation.

            shape ``batch_shape``

        :returns: shape ``batch_shape + (num_perms, state_dim, state_dim)``
        """
        ind = wrap_index(previous_perm_index)
        post_s_cond_init = self.logits[ind]
        post_s_cond_init = post_s_cond_init.unsqueeze(-3)
        post_s_cond_init = post_s_cond_init + observation_logits
        post_s_cond_init = post_s_cond_init - post_s_cond_init.logsumexp(axis=-1, keepdims=True)
        post_s_cond_init = _logmatmulexp(
            post_s_cond_init.float(),
            transition_logits.float()
        )
        return post_s_cond_init


class PermSelector(object):
    """
    This is an abstract class that is used to select permutations. The
    get_perm method is called in-line when sampling with PermutedDiscreteHMM.
    The get_perms method uses the get_perm method to compute all the
    permutations that would be chosen for all possible runs of data.

    See _perm_selector_template for an example of subclassing.
    """

    def __init__(self, possible_perms, save_history=False):
        n_perms, n_states = possible_perms.shape
        if not (possible_perms.long().sort(-1).values ==
                torch.arange(n_states, dtype=torch.long).expand(
                    (n_perms, n_states)
                )).all():
            raise ValueError("The input permutations are not permutations of "
                             "the integers [0, ..., n_states]")
        self.possible_perms = possible_perms
        self._calc_history = {}
        self._perm_history = []
        self.shape = None
        self.save_history = save_history

    @classmethod
    def manage_shape(cls, get_perm):
        """
        A decorator provided to flatten the batch dimensions of the input.
        :param get_perm: Permutation method to decorate.
        :return: Decorated method.
        """
        @wraps(get_perm)
        def _wrapper(self, *args, **kwargs):
            event_dims = kwargs.get("event_dims", 0)
            try:
                data_shape = args[0].shape
                shape = data_shape[:len(data_shape) - event_dims]
            except (AttributeError, IndexError):
                shape = None
            self_shape = getattr(self, "shape", None)
            if (self_shape is None) and (shape is not None):
                self.shape = shape
            data = args[0]
            if shape is not None:
                data = data.reshape((reduce(mul, self.shape, 1),) + data_shape[len(data_shape) - event_dims:])
            perm = get_perm(self, data, *args[1:], **kwargs)
            if shape is not None:
                perm = perm.reshape(shape + perm.shape[-1:])
            return perm
        return _wrapper

    @classmethod
    def manage_calc_history(cls, get_perm):
        """
        WARNING: This decorator changes the return signature of the decorated method.

        Given a method which returns a tuple whose first element is a permutation and whose
        second element is a dictionary containing ancillary information which is computed to
        compute the permutation, returns a method which returns only the permutation, while
        appending the ancillary information the self._calc_history

        :param get_perm: Method to compute the next permutation.
        :return: A method which returns only the permutation.

        ..seealso:: :py:meth:`perm_hmm.strategies.min_ent.MinEntropySelector.get_perm`
        """
        @wraps(get_perm)
        def _wrapper(self, *args, **kwargs):
            save_history = getattr(self, "save_history", False)
            retval = get_perm(self, *args, **kwargs)
            perm, calc_history = retval
            if save_history:
                for k, v in calc_history.items():
                    try:
                        self._calc_history[k].append(v)
                    except KeyError:
                        self._calc_history[k] = [v]
            return perm
        return _wrapper

    @classmethod
    def manage_perm_history(cls, get_perm):
        """
        Appends the permutation to self._perm_history.
        :param get_perm: Method to get the next permutation. Should return only a permutation.
        :return: Same method.
        """
        @wraps(get_perm)
        def _wrapper(self, *args, **kwargs):
            perm = get_perm(self, *args, **kwargs)
            self._perm_history.append(perm)
            return perm
        return _wrapper

    @property
    def perm_history(self):
        if len(self._perm_history) == 0:
            return torch.Tensor()
        else:
            try:
                toret = torch.stack(self._perm_history, dim=-2)
            except RuntimeError:
                return self._perm_history
            return toret

    @perm_history.setter
    def perm_history(self, val):
        self._perm_history = val

    @perm_history.deleter
    def perm_history(self):
        del self._perm_history

    @property
    def calc_history(self):
        if len(self._calc_history) == 0:
            return self._calc_history
        if any([len(v) == 0 for v in self._calc_history.values()]):
            return self._calc_history
        if self.shape is None:
            return self._calc_history
        try:
            return {k: torch.stack([x.reshape(self.shape + x.shape[1:]) for x in v], dim=-v[0].ndim) for k, v in self._calc_history.items()}
        except RuntimeError:
            return self._calc_history

    @calc_history.setter
    def calc_history(self, val):
        self._calc_history = val

    @calc_history.deleter
    def calc_history(self):
        del self._calc_history

    def get_perm(self, data: torch.Tensor, shape=()):
        """
        Takes a (vectorized) input of data from a single time step,
        and returns a (correspondingly shaped) permutation.
        :param torch.Tensor data: Data from the HMM.
            shape ``sample_shape + batch_shape + hmm.observation_dist.event_shape``
        :param save_history: A flag indicating whether or not to
            save the history of the computation involved to produce the
            permutations. The function shouldn't return anything
            different even if this flag is true, but the history should be
            available in the .history attribute at the end of the run.
        :return: The permutation to be applied at the next time step.
            shape ``(n_batches, n_states)``
        """
        raise NotImplementedError

    def reset(self, save_history=False):
        self.shape = None
        self._perm_history = []
        self.save_history = save_history
        self._calc_history = {}

    def get_perms(self, data, time_dim=-1):
        r"""
        Given a run of data, returns the permutations which would be applied.

        This should be used to precompute the permutations for a given model
        and given data sequence.

        :param torch.Tensor data: float.
            The sequence of data to compute the optimal permutations for

                shape ``batch_shape + (time_dim,)``

        :returns: A :py:class:`torch.Tensor` type :py:class:`int`
            containing the optimal permutations to have applied.

                shape ``batch_shape + (time_dim, num_states)``
        """
        d_shape = data.shape
        m = len(d_shape)
        if time_dim < 0:
            obs_event_dim = -(time_dim + 1)
        else:
            obs_event_dim = m - (time_dim + 1)
        shape = d_shape[:m - obs_event_dim]
        max_t = shape[-1]
        perms = []
        for i in range(max_t):
            perms.append(self.get_perm(
                data[(..., i) + (
                    slice(None),) * obs_event_dim],
            ))
        perms = torch.stack(perms, -2)


class MinEntropyPolicy(PermSelector):
    """
    A strategy for selecting permutations by choosing the one which gives the minimum
    expected posterior entropy of the initial state distribution given the
    past data and the next step of data, as yet unseen.

    """

    def __init__(self, possible_perms, hmm, save_history=False):
        # TODO: Fix this class to work with heterogeneous hmms

        super().__init__(possible_perms, save_history=save_history)
        self.hmm = hmm
        self.step = 0
        n_perms = len(possible_perms)
        self.prior_log_inits = \
            BayesInitialDistribution(self.hmm.initial_logits.clone().detach())
        """
        a :py:class:`BayesInitialDistribution`. Used to compute posterior
        initial state distributions.
        """

        self.prior_log_current = \
            BayesCurrentDistribution(
                self.hmm.initial_logits.clone().detach().repeat(n_perms, 1)
            )
        r"""
        a :py:class:`BayesCurrentDistribution`. Used to compute
        distributions of the form :math:`p(s_n|y^{i-1})`.
        """

        prior_log_cur_cond_init = \
            (torch.eye(len(self.hmm.initial_logits)) + ZERO).log()
        prior_log_cur_cond_init -= \
            prior_log_cur_cond_init.logsumexp(axis=-1, keepdim=True)
        self.prior_log_cur_cond_init = \
            BayesCurrentCondInitialDistribution(
                prior_log_cur_cond_init.repeat(n_perms, 1, 1)
            )

    @property
    def reverse_perm_dict(self):
        return {
            tuple(val.tolist()): torch.tensor(key, dtype=torch.long)
            for key, val in enumerate(self.possible_perms)
        }

    @reverse_perm_dict.setter
    def reverse_perm_dict(self, val):
        self.possible_perms = torch.stack(tuple(val.values()))

    def to_perm_index(self, perm):
        """
        Dualizes a permutation to its index in the :attr:`possible_perms`
        array.

        :param torch.Tensor perm: int. The perm to convert to an index.

            shape ``batch_shape + (state_dim,)``

        :returns: :py:class:`torch.Tensor`, int.

            shape ``batch_shape``
        """
        shape = perm.shape
        if len(shape) == 1:
            return self.reverse_perm_dict[tuple(perm.tolist())]
        flat_shape = (reduce(mul, shape[:-1]),) + shape[-1:]
        re_perm = perm.reshape(flat_shape)
        pi = torch.empty(flat_shape[:-1], dtype=torch.long)
        for i in range(re_perm.shape[0]):
            pi[i] = self.reverse_perm_dict[tuple(re_perm[i].tolist())]
        pi.reshape(shape[:-1])
        return pi

    def reset(self, save_history=False):
        """
        Resets the policy.
        """
        super().reset(save_history=save_history)
        n_perms = len(self.possible_perms)
        self.prior_log_inits.logits = self.hmm.initial_logits.clone().detach()
        self.prior_log_current.logits = \
            self.hmm.initial_logits.clone().detach().repeat(n_perms, 1)
        log_state_cond_initial_dist = \
            (torch.eye(len(self.hmm.initial_logits)) + ZERO).log()
        log_state_cond_initial_dist -= \
            log_state_cond_initial_dist.logsumexp(axis=-1, keepdim=True)
        self.prior_log_cur_cond_init.logits = \
            log_state_cond_initial_dist.repeat(n_perms, 1, 1)
        self.step = 0

    def update_prior(self, val):
        """
        Given a new observation and the permutation applied last,
         updates all the distributions being tracked.

        :param torch.Tensor val: torch.float an observed data point.
            This is :math:`y_i`.

            shape ``batch_shape``

        """
        n_states = len(self.hmm.initial_logits)
        shape = val.shape
        if len(self._perm_history) == 0:
            total_batches = shape[0]
            self.prior_log_current.logits = \
                self.prior_log_current.logits.expand(total_batches, -1, -1)
            self.prior_log_cur_cond_init.logits = \
                self.prior_log_cur_cond_init.logits.expand(
                    total_batches, -1, -1, -1)
            prev_perm = torch.arange(n_states, dtype=int)
        else:
            prev_perm = self._perm_history[-1]
        prev_perm = prev_perm.expand((shape[0],) + (n_states,))
        prev_perm_index = self.to_perm_index(prev_perm)
        transition_logits = self.hmm.transition_logits[self.possible_perms]
        if len(self.hmm.observation_dist.batch_shape) >= 2:
            observation_logits = self.hmm.observation_dist.log_prob(
                val.unsqueeze(-1).unsqueeze(-1)
            )[..., self.step, :].unsqueeze(-2).unsqueeze(-2)
        else:
            observation_logits = \
                self.hmm.observation_dist.log_prob(
                    val.unsqueeze(-1)
                ).unsqueeze(-2).unsqueeze(-2)
        prior_s_cond_init = self.prior_log_cur_cond_init.logits
        post_log_initial_dist = \
            self.prior_log_inits.posterior(
                observation_logits,
                prior_s_cond_init
            )
        ind = wrap_index(prev_perm_index)
        self.prior_log_inits.logits = post_log_initial_dist[ind]
        post_log_state_dist = \
            self.prior_log_current.posterior(observation_logits,
                                             transition_logits, prev_perm_index)
        self.prior_log_current.logits = post_log_state_dist
        post_log_state_cond_initial_dist = \
            self.prior_log_cur_cond_init.posterior(observation_logits,
                                                   transition_logits,
                                                   prev_perm_index)
        self.prior_log_cur_cond_init.logits = post_log_state_cond_initial_dist

    def full_posterior(self):
        r"""
        Computes the distributions needed to compute the posterior conditional
        entropy, which depends on yet to be seen data.

        :returns: a :py:class:`PostYPostS0` object, containing
            log_post_y: the posterior distribution
            :math:`\log(p(y_i | y^{i-1}))`

            shape ``(n_outcomes, n_perms)``

            log_post_init: the posterior distribution
            :math:`\log(p(s_0| y_i, y^{i-1}))`.

            shape ``(n_outcomes, n_perms, state_dim)``

        .. seealso:: method :py:meth:`PermutedDiscreteHMM.expected_entropy`
        """
        possible_outputs = \
            self.hmm.observation_dist.enumerate_support(False) \
                .squeeze().unsqueeze(-1)
        observation_logits = \
            self.hmm.observation_dist.log_prob(
                possible_outputs,
            ).float().unsqueeze(-2).unsqueeze(-2)
        for x in range(len(self.prior_log_inits.logits.shape) - 1):
            observation_logits.unsqueeze_(-2)
        log_post_y = self.prior_log_current.posterior_y(observation_logits)
        log_post_init = \
            self.prior_log_inits.posterior(
                observation_logits,
                self.prior_log_cur_cond_init.logits,
            )
        return PostYPostS0(log_post_y, log_post_init)

    def expected_entropy(self, output_distributions=False):
        r"""
        Computes the expected conditional entropy for all the permutations.

        :param bool output_distributions: indicates whether to return the
            tensor of posterior log initial state distributions and posterior
            log y distributions along with the entropy.

        :returns: Either a torch.Tensor of shape ``(n_perms,)`` or a
            :py:class:`GenDistEntropy` object, containing as its leaves,
            log_dists.log_post_y: Posterior log y distribution, i.e.
            :math:`\log(p(y_i | y^{i-1}))`.

            shape ``(n_outcomes, n_perms)``

            log_dists.log_post_init: Posterior log initial state distribution.
            i.e. :math:`\log(p(s_0 | y_i, y^{i-1}))`

            shape ``(n_outcomes, n_perms, state_dim)``

            expected_entropy: expected posterior entropy for each possible
            permutation.
            shape ``(n_perms,)``

        .. seealso:: method :py:meth:`PermutedDiscreteHMM.full_posterior`
        """
        postyposts1 = self.full_posterior()
        plid = postyposts1.log_post_init
        plyd = postyposts1.log_post_y
        pliyd = plid + plyd.unsqueeze(-1)
        entropy = (-pliyd.exp() * plid).sum(axis=-1).sum(axis=0)
        if output_distributions:
            return GenDistEntropy(PostYPostS0(plyd, plid), entropy)
        return entropy

    def calculate_perm(self, data: torch.Tensor) -> (torch.Tensor, dict):
        r"""
        Given data, returns the permutation which should be applied to the HMM before the next step, based on a minimum
        posterior entropy heuristic.

        :param data: Data from the HMM, used to update the computed distributions.
        :return: A tuple. First element is
            perm: :py:class:`torch.Tensor`
                dtype :py:class:`int`,
                Next permutation to apply.

                shape ``batch_shape + (state_dim,)``

            Second element is a dict, containing keys
            b"dist_array": A
                :py:class:`torch.Tensor` containing :math:`\log(p(s_0|y^i))`

                shape ``batch_shape + (state_dim,)``

            b"entropy_array": A :py:class:`torch.Tensor`
                containing
                :math:`\operatorname{min}_{\sigma}H_\sigma(S_0|Y^i, y^{i-1})`

                shape ``batch_shape``

        """
        self.update_prior(data)
        entropy = self.expected_entropy()
        entropy_array, perm_index = entropy.min(dim=-1)
        perm = self.possible_perms[perm_index]
        self.step += 1
        return perm, {b"dist_array": self.prior_log_inits.logits.clone().detach(), b"entropy_array": entropy_array}

    def _calculate_dists(self, data, perms):
        shape = perms.shape[:-1]
        if not data.shape[:len(shape)] == shape:
            raise ValueError("Data and permutations must have same shape")
        try:
            _ = self.hmm.log_prob(data)
        except (ValueError, RuntimeError) as e:
            raise ValueError("Data does not have a compatible shape") from e
        lperms = [torch.tensor(x) for x in perms.tolist()]
        self.reset(save_history=True)
        for i in range(shape[-1]):
            perm = self.get_perm(data[(..., i) + (slice(None),)*(len(data.shape)-len(shape))])
            self._perm_history = lperms[:i]
        return self.calc_history[b"dist_array"]


def simple_hmm():
    observation_probs = torch.tensor([.5, 1])
    observation_dist = dist.Bernoulli(observation_probs)
    possible_perms = torch.tensor([[0, 1], [1, 0], [0, 1]], dtype=int)
    transition_logits = torch.tensor([[1 - ZERO, ZERO], [.5, .5]]).log()
    initial_logits = torch.tensor([.5, .5]).log()
    hmm = PermutedDiscreteHMM(initial_logits, transition_logits, observation_dist)
    return hmm, possible_perms


@pytest.mark.parametrize("hmm,possible_perms,num_steps", [
    simple_hmm() + (4,),
    (three_state_hmm(-3, -4), id_and_transpositions(3), 4),
    (three_state_hmm(-1, -4), id_and_transpositions(3), 4),
    (three_state_hmm(-5, -4), id_and_transpositions(3), 4),
    ])
def test_posterior_distributions(hmm, possible_perms, num_steps):
    num_states = hmm.initial_logits.shape[0]
    all_data = all_strings(num_steps, num_states)
    mes1 = MinEntropyPolicy(possible_perms, hmm, save_history=True)
    mes2 = MES(possible_perms, hmm, save_history=True)
    for j in range(num_steps):
        mes1.update_prior(all_data[..., j])
        ply, pli = mes1.full_posterior()
        mes2.belief_state = mes2.belief_state.bayes_update(all_data[..., j])
        pl = mes2.distributions_for_all_perms()
        ply2 = pl.logsumexp(-3).logsumexp(-2)
        ply2 = torch.tensor(np.moveaxis(ply2.numpy(), (-1, -2, -3), (-3, -1, -2)))
        pli2 = pl.logsumexp(-2)
        pli2 = pli2 - pli2.logsumexp(-2, keepdim=True)
        pli2 = torch.tensor(np.moveaxis(pli2.numpy(), (-1, -2, -3, -4), (-4, -1, -2, -3)))
        assert torch.allclose(ply.exp().double(), ply2.exp().double(), atol=1e-6)
        assert torch.allclose(pli.exp().double(), pli2.exp().double(), atol=1e-6)
        perm = mes2.calculate_perm_from_belief(return_dict=False)
        mes1._perm_history.append(perm)
        mes2.belief_state = mes2.belief_state.transition(perm.unsqueeze(-2))


@pytest.mark.parametrize("hmm,possible_perms,num_steps",[
    simple_hmm() + (4,),
    (three_state_hmm(-3, -4), id_and_transpositions(3), 4),
    (three_state_hmm(-1, -4), id_and_transpositions(3), 4),
    (three_state_hmm(-5, -4), id_and_transpositions(3), 4),
    ])
def test_posterior_entropy(hmm, possible_perms, num_steps):
    num_states = hmm.initial_logits.shape[0]
    all_data = all_strings(num_steps, num_states)
    mes1 = MinEntropyPolicy(possible_perms, hmm, save_history=True)
    mes2 = MES(possible_perms, hmm, save_history=True)
    for j in range(num_steps):
        mes1.update_prior(all_data[..., j])
        entropy1 = mes1.expected_entropy()
        mes2.belief_state = mes2.belief_state.bayes_update(all_data[..., j])
        entropy2 = mes2.cond_entropies_for_all_perms()
        assert torch.allclose(entropy1.double(), entropy2.double(), atol=1e-6)
        perm = mes2.calculate_perm_from_belief(return_dict=False)
        mes1._perm_history.append(perm)
        mes2.belief_state = mes2.belief_state.transition(perm.unsqueeze(-2))


# @pytest.mark.parametrize("hmm,possible_perms,num_steps",[
#     simple_hmm() + (4,),
#     (three_state_hmm(-3, -4), id_and_transpositions(3), 4),
#     (three_state_hmm(-1, -4), id_and_transpositions(3), 4),
#     (three_state_hmm(-5, -4), id_and_transpositions(3), 4),
#     ])
# def test_posterior_perms(hmm, possible_perms, num_steps):
#     num_states = hmm.initial_logits.shape[0]
#     all_data = all_strings(num_steps, num_states)
#     mes1 = MinEntropyPolicy(possible_perms, hmm, save_history=True)
#     mes2 = MES(possible_perms, hmm, save_history=True)
#     for j in range(num_steps):
#         perm1 = mes1.get_perm(all_data[..., j])
#         perm2 = mes2.get_perm(all_data[..., j])
#         assert torch.all(perm1 == perm2)


# This test fails because of numerical precision issues.
# @pytest.mark.parametrize("hmm,possible_perms,num_steps",[
#     simple_hmm() + (4,),
#     (three_state_hmm(-3, -4), id_and_transpositions(3), 4),
#     (three_state_hmm(-1, -4), id_and_transpositions(3), 4),
#     (three_state_hmm(-5, -4), id_and_transpositions(3), 4),
#     ])
# def test_min_ent_consistent(hmm, possible_perms, num_steps):
#     all_data = all_strings(num_steps)
#     mes1 = min_ent.MinEntropyPolicy(possible_perms, hmm, save_history=True)
#     mes2 = min_ent_again.MinimumEntropyPolicy(possible_perms, hmm, save_history=True)
#     all_perms_1 = mes1.get_perms(all_data)
#     all_perms_2 = mes2.get_perms(all_data)
#     e1 = mes1.calc_history[b"entropy_array"]
#     e2 = mes2.calc_history[b"entropy"]
#     assert (all_perms_1 == all_perms_2).all()
#     assert torch.stack(e2).T.allclose(e1, atol=1e-6)
