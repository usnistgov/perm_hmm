"""This module includes a few utility functions.
"""
from functools import reduce
from operator import mul

import torch
import numpy as np
from scipy.special import logsumexp, expm1, log1p

ZERO = 10**(-14)


def bin_ent(logits_tensor):
    """Computes the binary entropy of a tensor of independent log probabilities.

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
    """Computes the entropy of a distribution.

    :param torch.Tensor log_dist: log of a distribution.
        The last axis should logsumexp to 0.

        shape ``(..., n_outcomes_of_distribution)``

    :returns: Entropy of the input distributions.
    """
    return (log_dist.exp() * (-log_dist)).sum(-1)


def num_to_data(num, num_steps, base=2, dtype=float):
    """Turns an integer into a tensor containing its representation in a given base.

    Use this function to, for example, enumerate all possible binary outcomes.

    :param int num: The integer whose binary representation is output
    :param base: The base of the resulting strings.
    :param int num_steps: The size of the output tensor
    :param dtype: The data type of the output tensor.
    :returns: A :py:class:`torch.Tensor` of length ``num_steps``
    """
    rep_list = []
    while num > 0:
        rep_list.append(num % base)
        num //= base
    assert len(rep_list) <= num_steps
    while len(rep_list) < num_steps:
        rep_list.append(0)
    rep_list = rep_list[::-1]
    return torch.tensor(rep_list, dtype=dtype)


def all_strings(steps, base=2, dtype=float):
    r"""All strings of a given base.

    :param steps: Length of strings
    :param base: The base of the strings.
    :param dtype: The type of the resulting tensor.
    :return: All strings of given base, as a :py:class:`~torch.Tensor`.
    """
    return torch.stack([num_to_data(num, steps, base, dtype=dtype) for num in range(base**steps)])


def unsqueeze_to(x, total, target):
    """Unsqueezes a dimension-1 tensor to the :attr:`target` position,
    out of a total of :attr:`total` dimensions.

    Example::

        >>> x = torch.arange(5)
        >>> y = unsqueeze_to(x, 6, 2)
        >>> assert y.shape == (1, 1, 5, 1, 1, 1)

    :param torch.Tensor x:
    :param int total:
    :param int target:
    :returns:
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
    """Gives all transpositions for length n as a list of
    :py:class:`~torch.Tensor`.

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


def id_and_transpositions(n):
    r"""Identity and transpositions.

    Computes a list of all transposition permutations, and an identity
    permutation.

    :param n: Number of states.
    :return: Shape ``(n*(n-1)/2, n)``
    """
    return torch.stack([torch.arange(n)] + transpositions(n))


def first_nonzero(x, dim=-1):
    """The first nonzero elements along a dimension.

    If none, default to length along dim.

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
    return to_ret*bx.any(dim) + (~bx.any(dim)).int()*bx.shape[dim]


def indices(shape):
    """An implementation of `numpy.indices <https://numpy.org/doc/stable/reference/generated/numpy.indices.html>`_
    for torch.

    Always returns the "sparse" version.

    :param tuple shape:
    :return: A tuple of tensors, each of dimension (1,)*n + (shape[n],) + (1,)*(len(shape) - n - 1), where n is the position in
        the tuple.
    """
    l = len(shape)
    return tuple(torch.arange(shape[a]).reshape((1,)*a + (shape[a],) + (1,)*(l-a-1)) for a in range(l))


def index_to_tuple(index, axis):
    """Given a tensor x which contains the indices into another tensor y, constructs the
    tuple to pass as y[index_to_tuple(x, axis)], where axis is the axis which x indexes.

    :param torch.Tensor index: An integer tensor whose elements can be interpreted as indices into another tensor.
    :param int axis: The axis which ``index`` indexes into.
    :return: A tuple of tensors which can be broadcast to shape ``index.shape``
    """
    shape = index.shape
    x = indices(shape)
    return x[:axis] + (index,) + x[axis:]


def log_tvd(lps1, lps2):
    r"""Log of `total variation distance`_, between two arrays with last axis
    containing the probabilities for all possible outcomes.

    .. math::

        \log(1/2 \sum_{x \in \mathcal{X}}|\mathbb{P}_1(x) - \mathbb{P}_2(x)|)

    where :math:`\mathcal{X}` is the space of possible outcomes.

    .. _`total variation distance`: https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures

    :param lps1:
    :param lps2:
    :return:
    """
    signs = np.sign(lps1 - lps2)
    retval = logsumexp(
        logsumexp(np.stack([lps1, lps2], axis=0), axis=0, b=np.stack([signs, -signs], axis=0)),
        axis=-1
    ) - np.log(2)
    return retval


def log1mexp(lp):
    r"""Log 1 minus exp of argument.

    .. math::

        \log(1-\exp(x))

    This is used to take :math:`1-p` for an argument in log space.

    :param lp:
    :return:
    """
    if lp > np.log(.5):
        return log1p(-np.exp(lp))
    return np.log(-expm1(lp))


def flatten_batch_dims(data, event_dims=0):
    """Flattens batch dimensions of data, and returns the batch shape.

    Data is assumed to be of shape batch_shape + event_shape. Pass
    len(event_shape) for the argument event_dims to delineate the event
    dimensions from the batch dimensions.

    :param data: The data to flatten.
    :param event_dims: The number of event dimensions. Defaults to 0.
    :return: The data with the batch dimensions flattened.
    :raises: Warning if the data shape does not match the previously
        recorded shape.
    """
    data_shape = data.shape
    shape = data_shape[:len(data_shape) - event_dims]
    batch_len = reduce(mul, shape, 1)
    event_shape = data_shape[len(data_shape) - event_dims:]
    data = data.reshape((batch_len,) + event_shape)
    return data, shape


def perm_idxs_from_perms(possible_perms, perms):
    idxs = (perms.unsqueeze(-2) == possible_perms).all(-1)  # type: torch.Tensor
    if (idxs.sum(-1) != 1).any():
        raise ValueError("Invalid permutations. Either the possible perms"
                         "contains duplicates, or there was a perm passed "
                         "that was not a possible perm.")
    return idxs.max(-1)[1]


def kl_divergence(lp1, lp2, axis=-1):
    mh1 = np.exp(logsumexp(np.log(-lp1) + lp1, axis=axis))
    mch = np.exp(logsumexp(np.log(-lp2) + lp1, axis=axis))
    return mch - mh1