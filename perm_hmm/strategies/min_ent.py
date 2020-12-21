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
from perm_hmm.hmms import SampleableDiscreteHMM, random_hmm
from perm_hmm.return_types import HMMOutput, PostYPostS0, GenDistEntropy, \
    PermWithHistory, PHMMOutHistory, PermHMMOutput
from perm_hmm.util import ZERO, wrap_index, id_and_transpositions
import copy
from selector import PermSelector


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


class MinEntropySelector(PermSelector):

    def __init__(self, possible_perms, hmm, calibrated=False):
        super().__init__(possible_perms)
        self.hmm = hmm
        self.calibrated = calibrated
        # A flag to guarantee that the model is trained.
        # The training should take place before it is used for
        # selecting permutations.
        self.reverse_perm_dict = {
            tuple(val.tolist()): torch.tensor(key, dtype=torch.long)
            for key, val in enumerate(self.possible_perms)
        }
        """
        The reverse of the permutations.
        """
        self.prior_log_inits = None
        self.prior_log_cur_cond_init = None
        self.prior_log_current = None
        self.previous_perm = None
        if self.calibrated:
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
        self._history = {
            "dist_array": [],
            "entropy_array": [],
        }
        self.shape = None

    @property
    def history(self):
        try:
            da = self._history["dist_array"]
            ea = self._history["entropy_array"]
        except KeyError:
            return self._history
        if (len(da) == 0) or (len(ea) == 0):
            return self._history
        if self.shape is None:
            return self._history
        try:
            return {k: torch.stack([x.reshape(self.shape) for x in v]) for k, v in self._history.items()}
        except RuntimeError:
            return self._history

    @history.setter
    def history(self, val):
        self._history = val

    @history.deleter
    def history(self):
        del self._history

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

    def reset(self):
        """
        Resets the selector.
        """
        if not self.calibrated:
            raise ValueError("Model is not yet calibrated.")
        n_perms = len(self.possible_perms)
        n_states = len(self.hmm.initial_logits)
        self.prior_log_inits.logits = self.hmm.initial_logits.clone().detach()
        self.prior_log_current.logits = \
            self.hmm.initial_logits.clone().detach().repeat(n_perms, 1)
        log_state_cond_initial_dist = \
            (torch.eye(len(self.hmm.initial_logits)) + ZERO).log()
        log_state_cond_initial_dist -= \
            log_state_cond_initial_dist.logsumexp(axis=-1, keepdim=True)
        self.prior_log_cur_cond_init.logits = \
            log_state_cond_initial_dist.repeat(n_perms, 1, 1)
        self.previous_perm = None
        self.shape = None
        self._history = {
            "dist_array": [],
            "entropy_array": [],
        }

    def update_prior(self, val):
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
        if self.previous_perm is None:
            n_states = len(self.hmm.initial_logits)
            shape = val.shape
            shape = shape[:len(shape) - self.hmm.observation_dist.event_dim]
            total_batches = reduce(mul, shape, 1)
            identity_perm = torch.arange(n_states, dtype=int)
            identity_perm = identity_perm.expand((total_batches,) + (n_states,))
            self.previous_perm = identity_perm
            self.shape = shape
        val = val.reshape((reduce(mul, self.shape, 1),) + self.hmm.observation_dist.event_shape)
        prev_perm_index = self.to_perm_index(self.previous_perm)
        transition_logits = self.hmm.transition_logits[self.possible_perms]
        observation_logits = \
            self.hmm.observation_dist.log_prob(
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

    def perm(self, data, save_history=False):
        self.update_prior(data)
        if save_history:
            self._history["dist_array"].append(self.prior_log_inits.logits)
        # total_batches = data.shape[0]
        # n_perms = self.possible_perms.shape[0]
        # entropy = self.expected_entropy().expand(total_batches, n_perms)
        entropy = self.expected_entropy()
        if save_history:
            entropy_array, perm_index = entropy.min(dim=-1)
            self._history["entropy_array"].append(entropy_array)
        else:
            perm_index = entropy.argmin(dim=-1)
        perm = self.possible_perms[perm_index]
        return perm

    def get_perms(self, data, save_history=False):
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
        d_shape = data.shape
        shape = d_shape[:len(d_shape)-self.hmm.observation_dist.event_dim]
        max_t = shape[-1]
        perms = torch.zeros(shape + (len(self.initial_logits),), dtype=int)
        self.reset()
        for i in range(max_t):
            perms[..., i, :] = self.perm(
                data[(..., i) + (slice(None),)*self.hmm.observation_dist.event_dim],
                save_history=save_history,
            )
        if save_history:
            return PermWithHistory(
                perms,
                self.history,
            )
        else:
            return perms
