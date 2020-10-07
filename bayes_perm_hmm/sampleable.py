"""
An adaptation of the `pyro.distributions.DiscreteHMM`_ class.

The additions are to the log_prob method (which is incorrect as written in the
pyro package), and the ability to sample from the model, functionality which is
not included in the `pyro`_ model.

.. _pyro.distributions.DiscreteHMM: http://docs.pyro.ai/en/stable/distributions.html?#pyro.distributions.DiscreteHMM
.. _pyro: http://docs.pyro.ai/en/stable/
"""

from operator import mul
from functools import reduce

import torch
import pyro
import pyro.distributions as dist
from pyro.distributions.hmm import DiscreteHMM
from pyro.distributions.hmm import _sequential_logmatmulexp
from pyro.distributions.util import broadcast_shape

from bayes_perm_hmm.return_types import HMMOutput


class SampleableDiscreteHMM(DiscreteHMM):
    """
    A discrete hidden Markov model which generates data.

    A mixin which adds a correct log_prob method, a vectorized sample method,
    and a method to compute the posterior log initial state distribution.
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

    def posterior_log_initial_state_dist(self, value):
        """
        Computes the posterior log initial state distribution.

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
        elif len(self.transition_logits.shape) == 3:
            result = self.transition_logits[:-1, :, :] + tail
            result = _sequential_logmatmulexp(result)
            result = result.logsumexp(-1)
            result = self.initial_logits + head + result
            result = result - result.logsumexp(-1, keepdim=True)
        else:
            raise ValueError('Wrong size for transition matrices')
        return result

    def parameters(self):
        """
        A parameters method to fit into the torch framework.
        :return: A list containing the initial log probs, the log transition
        probs, and the params which describe the observation distribution.
        """
        return \
            [
                self.initial_logits,
                self.transition_logits,
                self.observation_dist._param
            ]

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
            distribution.

            Otherwise all elements of
            :attr:`sample_shape` are interpreted as batch dimensions, and the
            time dimension of the model is always used.
            So
            ``batch_shape = sample_shape + self.batch_shape``,
            ``time_length = self.transition_logits.shape[-3]``

        :returns: ``batch_shape`` number of samples, each of length ``time_dim``

        :raises ValueError: if the model shape does not broadcast to the
            sample shape.
        """
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
        tmats = self.transition_logits.expand(
            flat_shape + self.transition_logits.shape[-2:]
        ).exp()
        b = self.observation_dist.batch_shape
        b_shape = broadcast_shape(shape[:-1], b[:-1])
        k = self.observation_dist._param.shape
        flat_params = \
            self.observation_dist._param.expand(
                b_shape + b[-1:] + (-1,)*(len(k)-len(b))
            ).reshape((total_batches,) + b[-1:] + (-1,)*(len(k)-len(b)))
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
                    flat_params[batch, states[batch, 0]]
                ),
            )
            for t in pyro.markov(range(1, time_shape[0])):
                states[batch, t] = pyro.sample(
                    "x_{}_{}".format(batch, t),
                    dist.Categorical(tmats[batch, t - 1, states[batch, t - 1]]),
                )
                observations[batch, t] = pyro.sample(
                    "y_{}_{}".format(batch, t),
                    type(self.observation_dist)(
                        flat_params[batch, states[batch, t]]
                    ),
                )
        states = states.reshape(shape)
        observations = observations.reshape(shape + self.observation_dist.event_shape)
        if (self.event_shape[0] == 1) and (sample_shape == ()):
            states.squeeze_(0)
            observations.squeeze_(0)
        return HMMOutput(states, observations)

    def log_prob(self, value):
        """
        Computes the log likelihood of the given observations.

        :param value: observations to compute the log_prob of.

            shape ``(batch_shape, time_dim)``

        :returns: The log likelihoods of the values.

            shape ``batch_shape``
        """
        value = value.unsqueeze(-1 - self.observation_dist.event_dim)
        observation_logits = self.observation_dist.log_prob(value)
        result = self.transition_logits + observation_logits.unsqueeze(-1)
        result = _sequential_logmatmulexp(result)
        result = self.initial_logits + result.logsumexp(-1)
        result = result.logsumexp(-1)
        return result


def random_hmm(n):
    dirichlet = dist.Dirichlet(torch.ones(n) / n)
    initial_logits = (torch.ones(n) / n).log()
    transition_logits = dirichlet.sample((n,))
    observation_dist = dist.Bernoulli(torch.rand(n))
    return SampleableDiscreteHMM(initial_logits, transition_logits, observation_dist)


