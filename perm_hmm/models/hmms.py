"""
An adaptation of the `pyro.distributions.DiscreteHMM`_ class.

The additions are to the log_prob method (which is incorrect as written in the
pyro package), and the ability to sample from the model, functionality which is
not included in the `pyro`_ model.

.. _pyro.distributions.DiscreteHMM: https://docs.pyro.ai/en/stable/distributions.html?#pyro.distributions.DiscreteHMM
.. _pyro: https://docs.pyro.ai/en/stable/
"""
from operator import mul
from functools import reduce

import torch
import pyro
import pyro.distributions as dist
import pyro.distributions.hmm
from pyro.distributions.hmm import _sequential_logmatmulexp
from pyro.distributions.util import broadcast_shape

from perm_hmm.util import wrap_index
from perm_hmm.return_types import HMMOutput
from perm_hmm.policies.policy import PermPolicy


class DiscreteHMM(pyro.distributions.hmm.DiscreteHMM):
    """A discrete hidden Markov model that generates data.

    Adds a correct log_prob method, a vectorized sample method,
    and a method to compute the posterior log initial state distribution.

    """

    def __init__(self, initial_logits, transition_logits, observation_dist,
                 validate_args=None):
        """Initializes the HMM.

        Just passes to the superclass initialization
        method with a check for the presence of the ``_param`` attribute in the
        ``observation_dist``.

        :raises ValueError: If the :attr:`observation_dist` doesn't have a
            :attr:`.param` attribute.
        """
        if not hasattr(observation_dist, '_param'):
            raise ValueError("The observation distribution should have a "
                             "'._param' attribute. Try reencoding your "
                             "distribution as a pyro.distributions.Categorical "
                             "object.")
        super().__init__(initial_logits, transition_logits, observation_dist,
                         validate_args=validate_args)
        self.has_enumerate_support = self.observation_dist.has_enumerate_support

    def enumerate_support(self, expand=True):
        return self.observation_dist.enumerate_support(expand)

    def posterior_log_initial_state_dist(self, value):
        """Computes the posterior log initial state distribution.

        This computation is similar to the forward algorithm.

        :param torch.Tensor value: The observed data.

            shape ``(batch_shape, time_dim)``

        :returns: The posterior log initial state distribution.

            shape ``(batch_shape, state_dim)``

        :raises ValueError: if the transition matrices are of the wrong size.
        """
        if value.shape[-1] == 0:
            return self.initial_logits
        if value.shape[-1] == 1:
            observation_logits = self.observation_dist.log_prob(value)
            result = observation_logits + self.initial_logits
            result -= result.logsumexp(-1, keepdim=True)
            return result

        value = value.unsqueeze(-1 - self.observation_dist.event_dim)
        value = value.float()
        observation_logits = self.observation_dist.log_prob(value)
        head = observation_logits[..., 0, :]
        tail = observation_logits[..., 1:, :]
        tail = tail.unsqueeze(-2)
        if len(self.transition_logits.shape) == 2:
            result = self.transition_logits + tail
            result = _sequential_logmatmulexp(result)
            result = result.logsumexp(-1)
            result = self.initial_logits + head + result
            result = result - result.logsumexp(-1, keepdim=True)
        elif len(self.transition_logits.shape) >= 3:
            result = self.transition_logits[..., :-1, :, :] + tail
            result = _sequential_logmatmulexp(result)
            result = result.logsumexp(-1)
            result = self.initial_logits + head + result
            result = result - result.logsumexp(-1, keepdim=True)
        else:
            raise ValueError('Wrong size for transition matrices')
        return result

    def parameters(self):
        """A parameters method to fit into the torch framework.

        :return: A list containing the initial log probs, the log transition
            probs, and the params which describe the observation distribution.
        """
        return \
            [
                self.initial_logits,
                self.transition_logits,
                self.observation_dist._param
            ]

    def _nonevent_output_shape(self, sample_shape=()):
        duration = self.duration
        if duration is None:
            if sample_shape == ():
                time_shape = (1,)
            else:
                time_shape = sample_shape[-1:]
            shape = sample_shape[:-1] + self.batch_shape + time_shape
        else:
            time_shape = (duration,)
            shape = sample_shape + self.batch_shape + time_shape
        return shape

    def _flatten_batch(self, shape):
        time_shape = shape[-1:]
        total_batches = reduce(mul, shape[:-1], 1)
        flat_shape = (total_batches,) + time_shape
        tmats = self.transition_logits.exp().expand(
            shape + self.transition_logits.shape[-2:]
        ).reshape(flat_shape + self.transition_logits.shape[-2:])
        b = self.observation_dist.batch_shape
        b_shape = broadcast_shape(shape, b[:-1])
        k = self.observation_dist._param.shape
        flat_params = \
            self.observation_dist._param.expand(
                b_shape + b[-1:] + (-1,)*(len(k)-len(b))
            ).reshape(flat_shape + b[-1:] + (-1,)*(len(k)-len(b)))
        return flat_shape, tmats, flat_params

    def sample(self, sample_shape=()):
        """Sample from the distribution.

        WARNING: This method does not return the correct answer for HMMs with
        heterogeneous outputs.

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
            distribution, otherwise all elements of
            :attr:`sample_shape` are interpreted as batch dimensions, and the
            time dimension of the model is always used.
            So
            ``batch_shape = sample_shape + self.batch_shape``,
            ``time_length = self.transition_logits.shape[-3]``

        :returns: ``batch_shape`` number of samples, each of length ``time_dim``

        :raises ValueError: if the model shape does not broadcast to the
            sample shape.
        """
        shape = self._nonevent_output_shape(sample_shape)
        flat_shape, tmats, flat_params = self._flatten_batch(shape)
        total_batches, steps = flat_shape
        dtype = self.observation_dist.sample().dtype
        states = torch.empty(flat_shape, dtype=int)
        observations = \
            torch.empty(flat_shape + self.observation_dist.event_shape, dtype=dtype)
        with pyro.plate("batches", total_batches) as batch:
            states[batch, 0] = pyro.sample("x_{}_0".format(batch),
                dist.Categorical(self.initial_logits.exp()),
            )
            observations[batch, 0] = pyro.sample(
                "y_{}_0".format(batch),
                type(self.observation_dist)(
                    flat_params[batch, 0, states[batch, 0]]
                ),
            )
            for t in pyro.markov(range(1, steps)):
                states[batch, t] = pyro.sample(
                    "x_{}_{}".format(batch, t),
                    dist.Categorical(tmats[batch, t - 1, states[batch, t - 1]]),
                )
                observations[batch, t] = pyro.sample(
                    "y_{}_{}".format(batch, t),
                    type(self.observation_dist)(
                        flat_params[batch, t, states[batch, t]]
                    ),
                )
        states = states.reshape(shape)
        observations = observations.reshape(shape + self.observation_dist.event_shape)
        return HMMOutput(states, observations)

    def log_prob(self, value):
        """Computes the log likelihood of the given observations.

        :param value: observations to compute the log_prob of.

            shape ``(batch_shape, time_dim)``

        :returns: The log likelihoods of the values.

            shape ``batch_shape``

        This code is based on the code for :py:class:`pyro.distributions.hmm.DiscreteHMM`,
        the license for this is in the ``licenses/HMM_LICENSE.md``.
        """
        value = value.unsqueeze(-1 - self.observation_dist.event_dim).float()
        observation_logits = self.observation_dist.log_prob(value)
        result = self.transition_logits + observation_logits.unsqueeze(-1)
        result = _sequential_logmatmulexp(result)
        result = self.initial_logits + result.logsumexp(-1)
        result = result.logsumexp(-1)
        return result


class PermutedDiscreteHMM(DiscreteHMM):
    """An HMM that allows for the underlying states to be permuted during a run.
    """

    def __init__(self, initial_logits, transition_logits, observation_dist,
                 validate_args=None):
        """
        :param initial_logits: log of the initial distribution

            shape ``(state_dim,)``

        :param transition_logits: log of the transition probabilities

            shape ``(state_dim, state_dim)``

        :param observation_dist: The output distribution of the HMM. Last
            dimension of its ``batch_shape`` should be of size ``state_dim``
            See :py:class:`~pyro.distributions.DiscreteHMM` for details on
            shape restrictions.

        :raises ValueError: If the :attr:`observation_dist` does not have the
            :meth:`enumerate_support` method.
        """
        if not observation_dist.has_enumerate_support:
            raise ValueError("The observation distribution must have the "
                             ".enumerate_support method.")
        super().__init__(initial_logits, transition_logits, observation_dist,
                         validate_args=validate_args)

    @classmethod
    def from_hmm(cls, hmm: DiscreteHMM):
        return cls(hmm.initial_logits, hmm.transition_logits, hmm.observation_dist)

    def sample(self, sample_shape=(), perm_policy: PermPolicy = None):
        r"""
        This method allows us to sample from the HMM with a given
        ``PermPolicy``.

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
            distribution, otherwise all elements of
            :attr:`sample_shape` are interpreted as batch dimensions, and the
            time dimension of the model is always used.
            So
            ``batch_shape = sample_shape + self.batch_shape``,
            ``time_length = self.transition_logits.shape[-3]``
        :param perm_policy: A PermPolicy object, must implement
            .get_perm, which is a method which takes batched data
            of shape ``batch_shape``
            and returns a batched permutation of shape
            ``batch_shape + (num_states,)``.

        :returns: A :py:class:`HMMOutput` object, containing

            `.states`: :py:class:`torch.Tensor`, dtype :py:class:`int`.
                The states realized during the run.

                shape ``batch_shape + (time_dim,)``

            `.observations`: :py:class:`torch.Tensor`,
                dtype :py:class:`float`.
                The output observations.

                shape ``batch_shape + (time_dim,)``
        """
        if perm_policy is None:
            return super().sample(sample_shape)

        shape = self._nonevent_output_shape(sample_shape)
        flat_shape, tmats, flat_params = self._flatten_batch(shape)
        total_batches, steps = flat_shape
        dtype = self.observation_dist.sample().dtype
        states = torch.empty(flat_shape, dtype=int)
        observations = \
            torch.empty(
                flat_shape + self.observation_dist.event_shape, dtype=dtype
            )
        with pyro.plate("batches", total_batches) as batch:
            states[batch, 0] = pyro.sample(
                "x_{}_0".format(batch),
                dist.Categorical(self.initial_logits.exp().repeat(total_batches, 1)),
            )
            observations[batch, 0] = pyro.sample(
                "y_{}_0".format(batch),
                type(self.observation_dist)(
                    flat_params[batch, 0, states[batch, 0]]
                ),
            )
            for t in pyro.markov(range(1, flat_shape[-1])):
                shaped_o = observations[batch, t-1].reshape(shape[:-1] + self.observation_dist.event_shape)
                perm = perm_policy.get_perm(shaped_o, event_dims=self.observation_dist.event_dim).reshape(total_batches, len(self.initial_logits))
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
                        flat_params[batch, t, states[batch, t]]
                    ),
                )
            shaped_o = observations[batch, -1].reshape(shape[:-1] + self.observation_dist.event_shape)
            perm = perm_policy.get_perm(shaped_o, event_dims=self.observation_dist.event_dim).reshape(total_batches, len(self.initial_logits))
        states = states.reshape(shape)
        observations = observations.reshape(shape + self.observation_dist.event_shape)
        return HMMOutput(
            states,
            observations,
        )

    def expand_with_perm(self, perm):
        """Expands the model along the time dimension, according to a
        permutation.

        :param perm: The list of permutations to apply. Should be of shape
            ``batch_shape + (num_steps, num_states)``.
        :return: An HMM expanded along the time dimension.
        """
        batch_shape = perm.shape[:-1]
        t_logits = self.transition_logits.expand(
            batch_shape + self.transition_logits.shape[-2:]
        )
        t_logits = t_logits[wrap_index(perm, batch_shape=perm.shape[:-1])]
        return type(self)(self.initial_logits, t_logits, self.observation_dist)

    def posterior_log_initial_state_dist(self, data, perm=None):
        """The posterior log initial state distributions for the data, given the
        permutations applied.

        :param torch.Tensor data: Data to compute the posterior initial state
            distribution for
        :param torch.Tensor perm: Permutations that were applied.
        :return:
        """
        if perm is None:
            return super().posterior_log_initial_state_dist(data)
        else:
            batch_shape = perm.shape[:-1]
            if data.shape[:len(data.shape)-self.observation_dist.event_dim] != batch_shape:
                raise ValueError("Perms and data do not have the same batch shape.")
            return self.expand_with_perm(perm).posterior_log_initial_state_dist(data)

    def log_prob(self, data, perm=None):
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
            :py:meth:`~perm_hmm.models.hmms.DiscreteHMM.log_prob`
        """
        if perm is None:
            return super().log_prob(data)
        batch_shape = perm.shape[:-1]
        if data.shape[:len(data.shape)-self.observation_dist.event_dim] != batch_shape:
            raise ValueError("Perms and data do not have the same batch shape.")
        return self.expand_with_perm(perm).log_prob(data)


def random_hmm(n):
    """A utility for generating random HMMs.

    Creates a uniformly random HMM with Bernoulli output. This means that each
    row of the transition matrix is sampled from the Dirichlet distribution of
    equal concentrations, as well as the initial state distribution, while the
    output distributions have their "bright" probability drawn uniformly from
    the unit interval.
    .. seealso:: :py:meth:`~perm_hmm.models.hmms.random_phmm`

    :param int n: Number of states for the HMM
    :return: A DiscreteHMM with Bernoulli outputs.
    """
    dirichlet = dist.Dirichlet(torch.ones(n) / n)
    initial_logits = (torch.ones(n) / n).log()
    transition_logits = dirichlet.sample((n,)).log()
    observation_dist = dist.Bernoulli(torch.rand(n))
    return DiscreteHMM(initial_logits, transition_logits, observation_dist)


def random_phmm(n):
    """A utility for generating random PermutedDiscreteHMMs.

    Creates a uniformly random HMM with Bernoulli output. This means that each
    row of the transition matrix is sampled from the Dirichlet distribution of
    equal concentrations, as well as the initial state distribution, while the
    output distributions have their "bright" probability drawn uniformly from
    the unit interval.

    .. seealso:: :py:meth:`~perm_hmm.models.hmms.random_hmm`

    :param int n: Number of states for the HMM
    :return: A PermutedDiscreteHMM with Bernoulli outputs.
    """
    hmm = random_hmm(n)
    return PermutedDiscreteHMM.from_hmm(hmm)


class SkipFirstDiscreteHMM(pyro.distributions.hmm.DiscreteHMM):
    """The initial state does not output.
    """
    def __init__(self, initial_logits, transition_logits, observation_dist,
                 validate_args=None):
        """
        Initializes the HMM. Just passes to the superclass initialization
        method with a check for the presence of an attribute.

        :raises ValueError: If the :attr:`observation_dist` doesn't have a
            :attr:`.param` attribute.
        """
        if not hasattr(observation_dist, '_param'):
            raise ValueError("The observation distribution should have a "
                             "'._param' attribute. Try reencoding your "
                             "distribution as a pyro.distributions.Categorical "
                             "object.")
        super().__init__(initial_logits, transition_logits, observation_dist,
                         validate_args=validate_args)
        self.has_enumerate_support = self.observation_dist.has_enumerate_support

    def enumerate_support(self, expand=True):
        return self.observation_dist.enumerate_support(expand)

    def posterior_log_initial_state_dist(self, value):
        """Computes the posterior log initial state distribution.

        :param torch.Tensor value: The observed data.

            shape ``(batch_shape, time_dim)``

        :returns: The posterior log initial state distribution.

            shape ``(batch_shape, state_dim)``

        :raises ValueError: if the transition matrices are of the wrong size.
        """
        if value.shape[-1] == 0:
            return self.initial_logits

        value = value.unsqueeze(-1 - self.observation_dist.event_dim)
        value = value.float()
        ol = self.observation_dist.log_prob(value)
        ol = ol.unsqueeze(-2)
        result = self.transition_logits + ol
        result = _sequential_logmatmulexp(result)
        result = result.logsumexp(-1)
        result = self.initial_logits + result
        result = result - result.logsumexp(-1, keepdim=True)
        return result

    def _nonevent_output_shape(self, sample_shape=()):
        duration = self.duration
        if duration is None:
            if sample_shape == ():
                time_shape = (1,)
            else:
                time_shape = sample_shape[-1:]
            shape = sample_shape[:-1] + self.batch_shape + time_shape
        else:
            time_shape = (duration,)
            shape = sample_shape + self.batch_shape + time_shape
        return shape

    def _flatten_batch(self, shape):
        time_shape = shape[-1:]
        total_batches = reduce(mul, shape[:-1], 1)
        flat_shape = (total_batches,) + time_shape
        tmats = self.transition_logits.exp().expand(
            shape + self.transition_logits.shape[-2:]
        ).reshape(flat_shape + self.transition_logits.shape[-2:])
        b = self.observation_dist.batch_shape
        b_shape = broadcast_shape(shape, b[:-1])
        k = self.observation_dist._param.shape
        flat_params = \
            self.observation_dist._param.expand(
                b_shape + b[-1:] + (-1,)*(len(k)-len(b))
            ).reshape(flat_shape + b[-1:] + (-1,)*(len(k)-len(b)))
        return flat_shape, tmats, flat_params

    def sample(self, sample_shape=()):
        """
        Sample from the distribution.


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
            distribution, otherwise all elements of
            :attr:`sample_shape` are interpreted as batch dimensions, and the
            time dimension of the model is always used.
            So
            ``batch_shape = sample_shape + self.batch_shape``,
            ``time_length = self.transition_logits.shape[-3]``

        :returns: ``batch_shape`` number of samples, each of length ``time_dim``

        :raises ValueError: if the model shape does not broadcast to the
            sample shape.
        """
        shape = self._nonevent_output_shape(sample_shape)
        flat_shape, tmats, flat_params = self._flatten_batch(shape)
        total_batches, steps = flat_shape
        dtype = self.observation_dist.sample().dtype
        states = torch.empty(flat_shape[:-1] + (steps + 1,), dtype=int)
        observations = \
            torch.empty(flat_shape + self.observation_dist.event_shape, dtype=dtype)
        with pyro.plate("batches", total_batches) as batch:
            states[batch, 0] = pyro.sample("x_{}_0".format(batch),
                                           dist.Categorical(self.initial_logits.exp()),
                                           )
            for t in pyro.markov(range(1, steps+1)):
                states[batch, t] = pyro.sample(
                    "x_{}_{}".format(batch, t),
                    dist.Categorical(tmats[batch, t - 1, states[batch, t - 1]]),
                )
                observations[batch, t-1] = pyro.sample(
                    "y_{}_{}".format(batch, t-1),
                    type(self.observation_dist)(
                        flat_params[batch, t-1, states[batch, t]]
                    ),
                )
        states = states.reshape(shape[:-1] + (steps+1,))
        observations = observations.reshape(shape + self.observation_dist.event_shape)
        return HMMOutput(states, observations)


class SkipFirstPermutedDiscreteHMM(SkipFirstDiscreteHMM):
    """Allows for the underlying states to be permuted during a run.

    .. seealso:: :py:class:`~perm_hmm.models.hmms.PermutedDiscreteHMM`
    """

    def __init__(self, initial_logits, transition_logits, observation_dist,
                 validate_args=None):
        """
        :param initial_logits: log of the initial distribution

            shape ``(state_dim,)``

        :param transition_logits: log of the transition probabilities

            shape ``(state_dim, state_dim)``

        :param observation_dist: The output distribution of the HMM. Last
            dimension of its ``batch_shape`` should be of size ``state_dim``
            See :py:class:`pyro.distributions.DiscreteHMM` for details on
            shape restrictions.

        :raises ValueError: If the :attr:`observation_dist` does not have the
            :meth:`enumerate_support` method.
        """
        if not observation_dist.has_enumerate_support:
            raise ValueError("The observation distribution must have the "
                             ".enumerate_support method.")
        super().__init__(initial_logits, transition_logits, observation_dist,
                         validate_args=validate_args)

    @classmethod
    def from_hmm(cls, hmm: SkipFirstDiscreteHMM):
        return cls(hmm.initial_logits, hmm.transition_logits, hmm.observation_dist)

    def sample(self, sample_shape=(), perm_policy: PermPolicy = None):
        r"""Samples from the distribution.

        Samples are generated using the ``perm_policy`` to select permutations
        of the underlying states at each step.

        The initial state does not have an output in this distribution.

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
            distribution, otherwise all elements of
            :attr:`sample_shape` are interpreted as batch dimensions, and the
            time dimension of the model is always used.
            So
            ``batch_shape = sample_shape + self.batch_shape``,
            ``time_length = self.transition_logits.shape[-3]``
        :param perm_policy: A PermPolicy object, must implement
            .get_perm, which is a method which takes batched data
            of shape ``batch_shape``
            and returns a batched permutation of shape
            ``batch_shape + (num_states,)``.

        :returns: A :py:class:`HMMOutput` object, containing

            `.states`: :py:class:`torch.Tensor`, dtype :py:class:`int`.
                The states realized during the run.

                shape ``batch_shape + (time_dim,)``

            `.observations`: :py:class:`torch.Tensor`,
                dtype :py:class:`float`.
                The output observations.

                shape ``batch_shape + (time_dim,)``
        """
        if perm_policy is None:
            return super().sample(sample_shape)

        shape = self._nonevent_output_shape(sample_shape)
        flat_shape, tmats, flat_params = self._flatten_batch(shape)
        total_batches, steps = flat_shape
        dtype = self.observation_dist.sample().dtype
        states = torch.empty(flat_shape[:-1] + (flat_shape[-1] + 1,), dtype=int)
        observations = \
            torch.empty(
                flat_shape + self.observation_dist.event_shape, dtype=dtype
            )
        with pyro.plate("batches", total_batches) as batch:
            states[batch, 0] = pyro.sample(
                "x_{}_0".format(batch),
                dist.Categorical(self.initial_logits.exp().repeat(total_batches, 1)),
            )
            perm = torch.arange(len(self.initial_logits)).expand(
                total_batches,
                -1,
            )
            for t in pyro.markov(range(1, flat_shape[-1]+1)):
                states[batch, t] = pyro.sample(
                    "x_{}_{}".format(batch, t),
                    dist.Categorical(
                        tmats[batch, t-1][
                            wrap_index(perm, perm.shape[:-1])
                        ][batch, states[batch, t-1]],
                    ),
                )
                observations[batch, t-1] = pyro.sample(
                    "y_{}_{}".format(batch, t-1),
                    type(self.observation_dist)(
                        flat_params[batch, t-1, states[batch, t]]
                    ),
                )
                shaped_o = observations[batch, t-1].reshape(
                    shape[:-1] + self.observation_dist.event_shape
                )  # Shape the observation before passing to perm, so that the
                # perms have the right shape in the perm_history later.
                perm = perm_policy.get_perm(
                    shaped_o,
                    event_dims=self.observation_dist.event_dim
                ).reshape(total_batches, len(self.initial_logits))
        states = states.reshape(shape[:-1] + (shape[-1]+1,))
        observations = observations.reshape(shape + self.observation_dist.event_shape)
        return HMMOutput(
            states,
            observations,
        )

    def expand_with_perm(self, perm):
        # HACK: We use the convention that the last permutation acts after the last
        # data, so it's irrelevant. Therefore, throw out the last permutation.
        # On the other hand, the initial permutation is always the identity, so
        # attach that.
        perm = perm[..., :-1, :]
        batch_shape = perm.shape[:-1]
        num_states = perm.shape[-1]
        iden = torch.arange(num_states).expand(perm.shape[:-2] + (1, num_states))
        perm = torch.cat((iden, perm), dim=-2)
        batch_shape = batch_shape[:-1] + (batch_shape[-1] + 1,)
        t_logits = self.transition_logits.expand(
            batch_shape + self.transition_logits.shape[-2:]
        )
        t_logits = t_logits[wrap_index(perm, batch_shape=perm.shape[:-1])]
        return type(self)(self.initial_logits, t_logits, self.observation_dist)

    def posterior_log_initial_state_dist(self, data, perm=None):
        """The posterior log initial state distributions for the data, given the
        permutations applied.

        :param torch.Tensor data: Data to compute the posterior initial state
            distribution for
        :param torch.Tensor perm: Permutations that were applied.
        :return:
        """
        if perm is None:
            return super().posterior_log_initial_state_dist(data)
        else:
            batch_shape = perm.shape[:-1]
            if data.shape[:len(data.shape)-self.observation_dist.event_dim] != batch_shape:
                raise ValueError("Perms and data do not have the same batch shape.")
            return self.expand_with_perm(perm).posterior_log_initial_state_dist(data)

    def log_prob(self, data, perm=None):
        """Computes the log prob of a run, using the permutation sequence
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
            :py:meth:`perm_hmm.models.hmms.DiscreteHMM.log_prob`
        """
        if perm is None:
            return super().log_prob(data)
        batch_shape = perm.shape[:-1]
        if data.shape[:len(data.shape)-self.observation_dist.event_dim] != batch_shape:
            raise ValueError("Perms and data do not have the same batch shape.")
        return self.expand_with_perm(perm).log_prob(data)


class ExpandedHMM(SkipFirstPermutedDiscreteHMM):
    r"""
    HMM with outcomes :math:`\mathcal{Y}`, and state space
    :math:`\mathcal{S} \times \mathcal{Y}`, where :math:`\mathcal{S}` is the
    physical state space.
    """

    def lo_to_i(self, lo):
        r"""Get serial index from tuple index.

        :param tuple lo: 2-tuple, a pair of :math:`(l, o) \in \mathcal{S} \times
            \mathcal{Y}`
        :return: Serial index :math:`i`
        """
        odim = self.observation_dist.enumerate_support().shape[0]
        return lo[0]*odim + lo[1]

    def i_to_lo(self, i):
        r"""Get tuple index from serial

        :param int i:
        :return: 2-tuple, a pair of :math:`(l, o) \in \mathcal{S} \times
            \mathcal{Y}`
        """
        odim = self.observation_dist.enumerate_support().shape[0]
        return divmod(i, odim)
