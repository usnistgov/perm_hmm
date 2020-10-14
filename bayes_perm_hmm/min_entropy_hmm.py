"""
Implements the Bayesian feedback inference method.

Conditioned on the data seen thus far, computes the expected posterior entropy
of the initial state, given the yet to be seen next data point, in expectation.
This computation is done for each allowed permutation. Then minimizing the
computed quantity over permutations, we obtain the permutation to apply.

All distributions are in log space.
"""
from operator import mul
from functools import reduce
import torch
import pyro
import pyro.distributions as dist
from pyro.distributions import DiscreteHMM
from pyro.distributions.hmm import _logmatmulexp
from pyro.distributions.util import broadcast_shape
from bayes_perm_hmm.sampleable import SampleableDiscreteHMM, random_hmm
from bayes_perm_hmm.return_types import HMMOutput, PostYPostS0, GenDistEntropy, \
    MinEntHistory, PermWithHistory, MinEntHMMOutput, PermHMMOutput
from bayes_perm_hmm.util import ZERO, wrap_index, transpositions_and_identity
import copy


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
    Stores the prior log intitial state distribution.

    This data structure stores :math:`\log(p(s_0 | y^{i-1}))`,
    where :math:`y^{i-1}` is all the data that has been seen so far.

    :param torch.Tensor logits: shape `` batch_shape + (state_dim,)``

    .. seealso:: Instantiated in :py:class:`PermutedDiscreteHMM`
    """

    def posterior(self, observation_logits, prior_s_cond_init):
        r"""
        Given a set of logits for the newly observed data and the distribution
        of the previous state conditional on the initial state, computes the
        posterior initial state distibution.

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
        return \
            (self.logits.unsqueeze(-2) +
                observation_logits).logsumexp(-1).squeeze(-1)


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


class PermutedDiscreteHMM(SampleableDiscreteHMM):
    """
    Computes minimal expected posterior entropy pemutations as new data is
    observed.
    """

    def __init__(self, initial_logits, transition_logits, observation_dist,
                 possible_perms: torch.Tensor, validate_args=None):
        """
        :param initial_logits: log of the initial distribution

            shape ``(state_dim,)``

        :param transition_logits: log of the transition probabilities

            shape ``(state_dim, state_dim)``

        :param observation_dist: The output distribution of the HMM. Last
            dimension of its ``batch_shape`` should be of size ``state_dim``
            See :py:class:`pyro.distributions.DiscreteHMM` for details on
            shape restrictions.

        :param torch.Tensor possible_perms: int tensor.
            Contains the allowable permutations.

            shape ``(n_perms, state_dim)``

        :raises ValueError: If the :attr:`observation_dist` does not have the
            :meth:`enumerate_support` method.
        """
        if not observation_dist.has_enumerate_support:
            raise ValueError("The observation distribution must have the "
                             ".enumerate_support method.")
        super().__init__(initial_logits, transition_logits, observation_dist,
                         validate_args=validate_args)
        self.state_dim = len(initial_logits)
        """
        the number of states in the model.
        """
        n_perms = len(possible_perms)
        if not (possible_perms.long().sort(-1).values ==
                torch.arange(self.state_dim, dtype=torch.long).expand(
                    (n_perms, self.state_dim)
                )).all():
            raise ValueError("The input permutations are not permutations of "
                             "the integers [0, ..., state_dim]")
        self.possible_perms = possible_perms.long()
        """
        All allowable permutations. Should have shape ``(n_perms, state_dim)``,
        with ``possible_perms[i]`` being a permutation of the tensor
        ``[0, ..., state_dim-1]`` for all i.
        shape ``(n_perms, state_dim)``
        """
        self.reverse_perm_dict = {
            tuple(val.tolist()): torch.tensor(key, dtype=torch.long)
            for key, val in enumerate(self.possible_perms)
        }
        """
        The reverse of the permutations.
        """

        self.prior_log_inits = \
            BayesInitialDistribution(self.initial_logits.clone().detach())
        """
        a :py:class:`BayesInitialDistribution`. Used to compute posterior
        initial state distributions.
        """

        self.prior_log_current = \
            BayesCurrentDistribution(
                self.initial_logits.clone().detach().repeat(n_perms, 1)
            )
        r"""
        a :py:class:`BayesCurrentDistribution`. Used to compute
        distributions of the form :math:`p(s_n|y^{i-1})`.
        """

        prior_log_cur_cond_init = \
            (torch.eye(len(initial_logits)) + ZERO).log()
        prior_log_cur_cond_init -= \
            prior_log_cur_cond_init.logsumexp(axis=-1, keepdim=True)
        self.prior_log_cur_cond_init = \
            BayesCurrentCondInitialDistribution(
                prior_log_cur_cond_init.repeat(n_perms, 1, 1)
            )
        r"""
        a :py:class:`BayesCurrentCondInitialDistribution`.
        Used to compute distributions of the form :math:`p(s_n|s_0, y^{i-1})`.
        """

    @classmethod
    def from_hmm(cls, hmm: DiscreteHMM, possible_perms):
        return cls(hmm.initial_logits, hmm.transition_logits, hmm.observation_dist, possible_perms)

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

    def update_prior(self, val, previous_perm: torch.Tensor):
        """
        Given a new observation and the permutation applied last,
         updates all the distributions being tracked.

        :param torch.Tensor val: torch.float an observed data point.
            This is :math:`y_i`.

            shape ``batch_shape``

        :param torch.Tensor previous_perm: int
            The previous permutation applied.

            shape ``batch_shape + (state_dim,)``
        """
        prev_perm_index = self.to_perm_index(previous_perm)
        transition_logits = self.transition_logits[self.possible_perms]
        observation_logits = \
            self.observation_dist.log_prob(
                val.unsqueeze(-1)
            ).float().unsqueeze(-2).unsqueeze(-2)
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
            self.observation_dist.enumerate_support(False) \
                .squeeze().unsqueeze(-1)
        observation_logits = \
            self.observation_dist.log_prob(
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

        :param bool output_distributions: indicates whether or not to return the
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
        entropy = \
            ((-plid.exp() * plid).sum(axis=-1) * plyd.exp()).sum(axis=0)
        if output_distributions:
            return GenDistEntropy(PostYPostS0(plyd, plid), entropy)
        return entropy

    def get_perms(self, data):
        r"""
        Given a run of data, returns the posterior initial state distributions,
        the optimal permutations according to
        the bayesian heuristic, and the posterior entropies.

        This should be used to precompute the permutations for a given model
        and given data sequence.

        :param torch.Tensor data: float.
            The sequence of data to compute the optimal permutations for

                shape ``batch_shape + (time_dim,)``

        :returns: A :py:class:`PermWithHistory` object with leaves

            .optimal_perm: A :py:class:`torch.Tensor` type :py:class:`int`
            containing the optimal permutations to have applied.

                shape ``batch_shape + (time_dim,)``

            .history.partial_post_log_init_dists: A :py:class:`torch.Tensor`
            containing :math:`p(s_0|y^{i})` for all :math:`i`.

                shape ``batch_shape + (time_dim, state_dim)``

            .history.expected_entropy: A :py:class:`torch.Tensor` containing
            :math:`\operatorname{min}_{\sigma}H_\sigma(S_0|Y^i, y^{i-1})`
            for all :math:`i`.

                shape ``batch_shape + (time_dim,)``

        .. seealso:: method :py:meth:`PermutedDiscreteHMM.sample_min_entropy`
        """
        self.reset()
        data_shape = data.shape
        max_t = data_shape[-1]
        batch_shape = data_shape[:-1]
        if len(data_shape) != 1:
            self.prior_log_current.logits = \
                self.prior_log_current.logits.expand(batch_shape + (-1, -1))
            self.prior_log_cur_cond_init.logits = \
                self.prior_log_cur_cond_init.logits.expand(
                    batch_shape + (-1, -1, -1)
                )
        perm_index_array = torch.zeros(data_shape, dtype=int)
        dist_array = torch.zeros(data_shape + (self.state_dim,))
        entropy_array = torch.zeros_like(data)
        identity_perm = torch.arange(self.state_dim, dtype=int)
        identity_perm = identity_perm.expand(batch_shape + (self.state_dim,))
        self.update_prior(
            data[(..., 0) + (slice(None),)*self.observation_dist.event_dim],
            identity_perm,
        )
        for i in range(1, max_t):
            dist_array[..., i - 1, :] = self.prior_log_inits.logits
            entropy = self.expected_entropy()
            entropy_array[..., i-1], perm_index_array[..., i-1] = \
                entropy.min(dim=-1)
            self.update_prior(
                data[(..., i) + (slice(None),)*self.observation_dist.event_dim],
                self.possible_perms[perm_index_array[..., i - 1]],
            )
        dist_array[..., -1, :] = self.prior_log_inits.logits
        entropy = self.expected_entropy()
        entropy_array[..., -1], perm_index_array[..., -1] = \
            entropy.min(dim=-1)
        self.reset()
        return PermWithHistory(
            self.possible_perms[perm_index_array],
            MinEntHistory(
                dist_array,
                entropy_array,
            ),
        )

    def reset(self):
        """
        A helper method to reset the tracked distributions after a computation
        which involves changing them in place.
        """
        n_perms = len(self.possible_perms)
        self.prior_log_inits.logits = self.initial_logits.clone().detach()
        self.prior_log_current.logits = \
            self.initial_logits.clone().detach().repeat(n_perms, 1)
        log_state_cond_initial_dist = \
            (torch.eye(len(self.initial_logits)) + ZERO).log()
        log_state_cond_initial_dist -= \
            log_state_cond_initial_dist.logsumexp(axis=-1, keepdim=True)
        self.prior_log_cur_cond_init.logits = \
            log_state_cond_initial_dist.repeat(n_perms, 1, 1)

    def get_posterior(self, data, perm: torch.Tensor):
        """
        Computes the posterior log initial state distributions for a given
        set of data and a given set of permutation indices.

        :param torch.Tensor data: torch.float
            The run of data which was observed.

                shape ``batch_shape + (time_dim,)``

        :param torch.Tensor perm: int, The indices corresponding to the
            permutations which were applied during the creation of the data. Its
            .perm should have

                shape ``batch_shape + (time_dim,)``

        :returns: A :py:class:`torch.Tensor` torch.float
            the posterior
            log initial state distributions :math:`p(s_0 | y^i)`.

                shape ``batch_shape + (time_dim, state_dim)``

        """
        data_shape = data.shape
        max_t = data_shape[-1]
        batch_shape = data_shape[:-1]
        dist_array = torch.zeros(data_shape + (self.state_dim,))
        self.reset()
        identity_perm = torch.arange(self.state_dim, dtype=int)
        identity_perm = identity_perm.expand(batch_shape + (self.state_dim,))
        self.update_prior(
            data[(..., 0) + (slice(None),)*self.observation_dist.event_dim],
            identity_perm,
        )
        for i in range(1, max_t):
            dist_array[..., i - 1, :] = self.prior_log_inits.logits
            self.update_prior(
                data[(..., i) + (slice(None),)*self.observation_dist.event_dim],
                perm[..., i-1, :],
            )
        dist_array[..., -1, :] = self.prior_log_inits.logits
        self.reset()
        return dist_array

    def sample_min_entropy(self, sample_shape=(), save_history=True):
        r"""
        This method allows us to sample from the HMM with the minimum expected
        posterior entropy heuristic.

        :param tuple sample_shape: tuple of ints. If the model doesn't contain a
            time dimension, i.e. if :attr:`transition_logits` has only two
            dimensions, then the last element of :attr:`sample_shape` is taken
            to be the time dimension, and all others will be
            treated independently as a batch.
            So
            ``batch_shape = sample_shape[:-1] + self.batch_shape``,
            ``time_length = sample_shape[-1]``

            If :attr:`sample_shape` is the empty tuple and the model doesn't
            contain a time dimension, we just sample from the initial
            distribution.

            Otherwise all elements of
            :attr:`sample_shape` are interpreted as batch dimensions, and the
            time dimension of the model is always used.
            So
            ``batch_shape = sample_shape + self.batch_shape``,
            ``time_length = self.transition_logits.shape[-3]``

        :returns: A :py:class:`MinEntHMMOutput` object, containing

            `.sample.states`: :py:class:`torch.Tensor`, dtype :py:class:`int`.
                The states realized during the run.

                shape ``batch_shape + (time_dim,)``

            `.sample.observations`: :py:class:`torch.Tensor`,
                dtype :py:class:`float`.
                The output observations.

                shape ``batch_shape + (time_dim,)``

            `.bayesian.optimal_perm`: :py:class:`torch.Tensor`
                dtype :py:class:`int`,
                The optimal permutations to have applied during this sequence of
                observations.

                shape ``batch_shape + (time_dim, state_dim)``

            `.bayesian.history.partial_post_log_init_dists`: A
                :py:class:`torch.Tensor` containing :math:`\log(p(s_0|y^i))`
                for all :math:`i`.

                shape ``batch_shape + (time_dim, state_dim)``

            `.bayesian.history.expected_entropy`: A :py:class:`torch.Tensor`
                containing
                :math:`\operatorname{min}_{\sigma}H_\sigma(S_0|Y^i, y^{i-1})`
                for all :math:`i`.

                shape ``batch_shape + (time_dim,)``
        """
        self.reset()
        if self.event_shape[0] == 1:
            if sample_shape == ():
                time_shape = (1,)
            else:
                time_shape = sample_shape[-1:]
            shape = sample_shape[:-1] + self.batch_shape + time_shape
        else:
            time_shape = (self.event_shape[0],)
            shape = sample_shape + self.batch_shape + time_shape
        total_batches = reduce(mul, shape[:-1], 1)
        flat_shape = (total_batches,) + time_shape
        n_perms = len(self.possible_perms)

        tmats = \
            self.transition_logits.expand(
                flat_shape + self.transition_logits.shape[-2:]).exp()
        self.prior_log_current.logits = \
            self.prior_log_current.logits.expand(total_batches, -1, -1)
        self.prior_log_cur_cond_init.logits = \
            self.prior_log_cur_cond_init.logits.expand(
                total_batches, -1, -1, -1)
        b = self.observation_dist.batch_shape
        b_shape = broadcast_shape(shape[:-1], b[:-1])
        k = self.observation_dist._param.shape
        flat_params = \
            self.observation_dist._param.expand(
                b_shape + b[-1:] + (-1,)*(len(k)-len(b))
            ).reshape((total_batches,) + b[-1:] + (-1,)*(len(k)-len(b)))
        dtype = self.observation_dist.sample().dtype
        identity_perm = torch.arange(self.state_dim, dtype=int)
        identity_perm = \
            identity_perm.expand((total_batches,) + (self.state_dim,))

        perm_index = torch.zeros(flat_shape, dtype=int)
        if save_history:
            dist_array = torch.zeros(flat_shape + (self.state_dim,))
            entropy_array = torch.zeros(flat_shape)
        states = torch.empty(flat_shape, dtype=int)
        observations = \
            torch.empty(
                flat_shape + self.observation_dist.event_shape, dtype=dtype
            )

        with pyro.plate("batches", total_batches) as batch:
            states[batch, 0] = pyro.sample(
                "x_{}_0".format(batch),
                dist.Categorical(self.initial_logits.exp()),
            )
            observations[batch, 0] = pyro.sample(
                "y_{}_0".format(batch),
                type(self.observation_dist)(
                    flat_params[batch, states[batch, 0]]
                ),
            )
            self.update_prior(
                observations[batch, 0],
                identity_perm,
            )
            for t in pyro.markov(range(1, flat_shape[-1])):
                entropy = self.expected_entropy().expand(total_batches, n_perms)
                if save_history:
                    dist_array[batch, t - 1] = self.prior_log_inits.logits
                    entropy_array[batch, t-1], perm_index[batch, t-1] = \
                        entropy.min(dim=-1)
                else:
                    perm_index[batch, t-1] = entropy.argmin(dim=-1)
                perm = self.possible_perms[perm_index[batch, t-1]]
                states[batch, t] = pyro.sample(
                    "x_{}_{}".format(batch, t),
                    dist.Categorical(
                        tmats[batch, t-1][
                            wrap_index(perm, perm.shape[:-1])
                        ][batch, states[batch, t-1]],
                    ),
                )
                observations[batch, t] = pyro.sample(
                    "y_{}_{}".format(batch, t),
                    type(self.observation_dist)(
                        flat_params[batch, states[batch, t]]
                    ),
                )
                self.update_prior(
                    observations[batch, t],
                    self.possible_perms[perm_index[batch, t-1]]
                )
            entropy = self.expected_entropy().expand(total_batches, n_perms)
            if save_history:
                dist_array[batch, -1] = self.prior_log_inits.logits
                entropy_array[batch, -1], perm_index[batch, -1] = \
                    entropy.min(dim=-1)
            else:
                perm_index[batch, -1] = entropy.argmin(dim=-1)
        self.reset()
        states = states.reshape(shape)
        observations = observations.reshape(shape + self.observation_dist.event_shape)
        perm_index = perm_index.reshape(shape)
        if save_history:
            dist_array = dist_array.reshape(shape + (self.state_dim,))
            entropy_array = entropy_array.reshape(shape)
        if (self.event_shape[0] == 1) and (sample_shape == ()):
            states.squeeze_(0)
            observations.squeeze_(0)
            perm_index.squeeze_(0)
            if save_history:
                dist_array.squeeze_(0)
                entropy_array.squeeze_(0)
        if save_history:
            return MinEntHMMOutput(
                HMMOutput(states, observations),
                self.possible_perms[perm_index],
                MinEntHistory(
                    dist_array,
                    entropy_array,
                ),
            )
        else:
            return PermHMMOutput(
                HMMOutput(states, observations),
                self.possible_perms[perm_index]
            )

    def log_prob_with_perm(self, perm: torch.Tensor, data):
        """
        Computes the log prob of a run, using the permutation sequence
        that was applied to generate the data.

        :param torch.Tensor perm: int.
            The encoded permutations
            applied to the HMM to generate the data.

        :param torch.Tensor data: float.
            A tensor containing the data to compute the log_prob for.

        :returns: float :py:class:`torch.Tensor`.
            The log probability of the data under the model where the
            permutations encoded by perm is applied.

            shape ``perm.shape[:-1]``

        :raises ValueError: if :attr:`perm` and :attr:`data` are not compatible
            shapes.

        .. seealso:: Method
            :py:meth:`bayes_perm_hmm.sampleable.SampleableDiscreteHMM.log_prob`
        """
        batch_shape = perm.shape[:-1]
        if data.shape[:len(data.shape)-self.observation_dist.event_dim] != batch_shape:
            raise ValueError("Perms and data do not have the same batch shape.")
        t_logits = self.transition_logits.expand(
            batch_shape + self.transition_logits.shape[-2:]
        )
        t_logits = t_logits[wrap_index(perm, batch_shape=perm.shape[:-1])]
        return SampleableDiscreteHMM(self.initial_logits,
                                     t_logits,
                                     self.observation_dist).log_prob(data)

    def posterior_log_initial_state_dist(self, data, perm=None):
        """
        The posterior log initial state distributions for the data, given the
        permutations applied.
        :param torch.Tensor data: Data to compute the posterior initial state
        distribution for
        :param torch.Tensor perm: Permutations which were applied.
        :return:
        """
        if perm is None:
            return super().posterior_log_initial_state_dist(data)
        else:
            batch_shape = perm.shape[:-1]
            if data.shape[:len(data.shape)-self.observation_dist.event_dim] != batch_shape:
                raise ValueError("Perms and data do not have the same batch shape.")
            t_logits = self.transition_logits.expand(
                batch_shape + self.transition_logits.shape[-2:]
            )
            t_logits = t_logits[wrap_index(perm, batch_shape=perm.shape[:-1])]
            return SampleableDiscreteHMM(self.initial_logits,
                                         t_logits,
                                         self.observation_dist).posterior_log_initial_state_dist(data)


def random_phmm(n):
    hmm = random_hmm(n)
    return PermutedDiscreteHMM.from_hmm(hmm, transpositions_and_identity(n))

