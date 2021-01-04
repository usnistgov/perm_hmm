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
from pyro.distributions.hmm import _logmatmulexp
from perm_hmm.return_types import PostYPostS0, GenDistEntropy
from perm_hmm.util import ZERO, wrap_index
from perm_hmm.strategies.selector import PermSelector


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


class MinEntropySelector(PermSelector):
    """
    A strategy for selecting permutations by choosing the one which gives the minimum
    expected posterior entropy of the initial state distribution given a the
    past data and the next step of data, as yet unseen.

    """

    def __init__(self, possible_perms, hmm, save_history=False):
        # TODO: Fix this class to work with heterogeneous hmms

        super().__init__(possible_perms, save_history=save_history)
        self.hmm = hmm
        self.prior_log_inits = None
        self.prior_log_cur_cond_init = None
        self.prior_log_current = None
        self.time_step = 0
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
        Resets the selector.
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
        self.time_step = 0

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
            )[..., self.time_step, :].unsqueeze(-2).unsqueeze(-2)
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

    @PermSelector.manage_perm_history
    @PermSelector.manage_shape
    @PermSelector.manage_calc_history
    def get_perm(self, data, event_dims=0):
        r"""
        Given data, returns the permutation which should be applied to the HMM before the next step, based on a minimum
        posterior entropy heuristic.

        :param data: Data from the HMM, used to update the computed distributions.
        :param event_dims: The number of dimensions which should be interpreted as event dimensions.
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
        self.time_step += 1
        return perm, {b"dist_array": self.prior_log_inits.logits, b"entropy_array": entropy_array}

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
