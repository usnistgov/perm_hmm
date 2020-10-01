"""
This module includes a few utility functions.
"""

import torch
import torch.distributions as dist
from bayes_perm_hmm.sampleable import SampleableDiscreteHMM
from bayes_perm_hmm.min_entropy_hmm import PermutedDiscreteHMM
import copy


ZERO = 10**(-14)


def bin_ent(logits_tensor):
    """
    Computes the binary entropy of a tensor of independent log probabilities.

    :param torch.Tensor logits_tensor:
        A tensor of log independent Bernoulli parameters.
        All elements should be nonpositive.

        shape arbitrary.

    :returns: A tensor of the same shape containing the binary entropies
        corresponding to the inputs.
    """
    y = (1 - logits_tensor.exp())
    return -(y.log() * y + logits_tensor.exp() * logits_tensor)


def entropy(log_dist):
    """
    Computes the entropy of a distribution.

    :param torch.Tensor log_dist: log of a distribution.
        The last axis should logsumexp to 0.

        shape ``(..., n_outcomes_of_distribution)``

    :returns: Entropy of the input distributions.
    """
    return (log_dist.exp() * (-log_dist)).sum(-1)


def num_to_data(num, num_bins):
    """
    Turns an integer into a tensor containing its binary representation.

    Use this function to enumerate all possible binary outcomes.

    :param int num: The integer whose binary representation is output

    :param int num_bins: The size of the output tensor

    :returns: A :py:class:`torch.Tensor` of length ``num_bins``
    """
    return torch.tensor(
        list(map(int, ("{{0:0{}b}}".format(num_bins)).format(num))),
        dtype=torch.float,
    )


def unsqueeze_to(x, total, target):
    """
    Unsqueezes a dimension-1 tensor to the :attr:`target` position,
    out of a total of :attr:`total` dimensions.

    :param torch.Tensor x: A dimension-1 tensor.
    :param int total: The total desired number of dimensions.
    :param int target: The target dimension.
    :returns: A view of :attr:`x` with enough dimensions unsqueezed.
    """
    ones = total - target - 1
    twos = target
    r = x
    for i in range(twos):
        r.unsqueeze_(-2)
    for i in range(ones):
        r.unsqueeze_(-1)
    return r


def wrap_index(index: torch.Tensor, batch_shape=None):
    """
    Takes a tensor whose interpretation is as indices of another
    tensor of shape ``batch_shape + arbitrary`` and outputs
    a tuple of tensors with which we can slice into the desired tensor.

    Use this method when you want only the 'diagonal' elements of some other
    tensor.

    Example::

        >>> x = torch.rand((5, 10, 7))
        >>> v = x.argmin(dim=-1)
        >>> print((x.min(dim=-1)[0] == x[wrap_index(v)].all()))
        tensor(True)

    :param torch.Tensor index: dtype :py:class:`int`. The index we wish to use
        to slice into some other tensor
    :param tuple batch_shape: tuple of ints or a :py:class:`torch.Shape` object.
        The shape of the batch dimensions of the tensor we will slice into.
        If :class:`None`, defaults to `index.shape`.
    :return: tuple of tensors. Use it to slice into some other tensor of the
        right shape.
    """
    shape = index.shape
    if batch_shape is None:
        batch_shape = index.shape
    if batch_shape == ():
        return index
    l = len(shape)
    ind = \
        tuple(
            unsqueeze_to(torch.arange(batch_shape[x]), l, x)
            for x in range(len(batch_shape))
        )
    ind = ind + (index,)
    return ind


def transpositions(n):
    """
    Gives all transpositions for length n as a list of :class:`torch.Tensor`.
    :param int n: The number to compute for.
    :return: list of transpositions.
    """
    ts = []
    for i in range(n):
        for j in range(i):
            x = torch.arange(n)
            x[i] = j
            x[j] = i
            ts.append(x)
    return ts


def transpositions_and_identity(n):
    return torch.stack([torch.arange(n)] + transpositions(n))


def random_hmm(n):
    dirichlet = dist.Dirichlet(torch.ones(n) / n)
    initial_logits = (torch.ones(n) / n).log()
    transition_logits = dirichlet.sample((n,))
    observation_dist = dist.Bernoulli(torch.rand(n))
    return SampleableDiscreteHMM(initial_logits, transition_logits, observation_dist)


def add_permutations(hmm, permutations):
    il = hmm.initial_logits.clone()
    tl = hmm.transition_logits.clone()
    od = copy.deepcopy(hmm.observation_dist)
    pdh = PermutedDiscreteHMM(il, tl, od, permutations)
    return(pdh)


def random_phmm(n):
    hmm = random_hmm(n)
    return add_permutations(hmm, transpositions_and_identity(n))


def first_nonzero(x, dim=-1):
    """
    The first nonzero elements along a dimension. If none, default to -1.
    :param torch.Tensor x:
    :param int dim:
    :return: x reduced along dim.
    """
    s = x.shape
    if len(s) == 0:
        x = x.unsqueeze(-1)
    l = x.shape[dim]
    bx = x.bool()
    ix = bx.int()
    rl = torch.arange(l, 0, -1).view((l,)+(1,)*len(x.shape[dim:-1]))
    to_argmax = ix * rl
    to_ret = to_argmax.argmax(dim)
    return to_ret*bx.any(dim) - (~bx.any(dim)).int()


def indices(shape):
    """
    An implementation of `numpy.indices <https://numpy.org/doc/stable/reference/generated/numpy.indices.html>`_
    for torch. Always returns the "sparse" version.
    :param tuple shape:
    :return: A tuple of tensors, each of dimension (1,)*n + (shape[n],) + (1,)*(len(shape) - n - 1), where n is the position in
    the tuple.
    """
    l = len(shape)
    return tuple(torch.arange(shape[a]).reshape((1,)*a + (shape[a],) + (1,)*(l-a-1)) for a in range(l))


def index_to_tuple(index, axis):
    """
    Given a tensor x which contains the indices into another tensor y, constructs the
    tuple to pass as y[index_to_tuple(x, axis)], where axis is the axis which x indexes.
    :param torch.Tensor index: An integer tensor whose elements can be interpreted as indices into another tensor.
    :param int axis: The axis which ``index`` indexes into.
    :return: A tuple of tensors which can be broadcast to shape ``index.shape``
    """
    shape = index.shape
    l = len(shape)
    x = indices(shape)
    return x[:axis] + (index,) + x[axis:]