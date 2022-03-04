r"""Loss functions for the
:py:class:`~perm_hmm.postprocessing.ExactPostprocessor` and
:py:class:`~perm_hmm.postprocessing.EmpiricalPostprocessor` classes.
"""
import torch

from perm_hmm.util import ZERO


def log_zero_one(state, classification):
    r"""Log zero-one loss.

    Returns ``log(int(classification != state))``

    The log of zero is clipped to be ``np.log(perm_hmm.util.ZERO)``.
    """
    loss = classification != state
    floss = loss.float()
    floss[~loss] = ZERO
    log_loss = floss.log()
    return log_loss


def zero_one(state, classification):
    r"""Zero-one loss function.

    Returns ``classification != state``
    """
    return classification != state


def log_binary_zero_one(dark_state, bright_state):
    testing_states = torch.tensor([dark_state, bright_state], dtype=int)

    def _wrapper(state, classification):
        return log_zero_one(state, testing_states[classification.long()])
    return _wrapper


def binary_zero_one(dark_state, bright_state):
    r"""
    Makes the identification of 0 = ``dark_state``, 1 = ``bright_state`` then
    returns the zero-one loss function for that.
    """
    testing_states = torch.tensor([dark_state, bright_state], dtype=int)

    def _wrapper(state, classification):
        return zero_one(state, testing_states[classification.long()])
    return _wrapper


def mapped_log_zero_one(state, classification, alpha=None):
    r"""For when the state and classification are considered equal for some
    nontrivial mapping ``alpha``.

    Returns ``log_zero_one(alpha(state), alpha(classification))``.
    """
    if alpha is None:
        return log_zero_one(state, classification)
    else:
        return log_zero_one(alpha(state), alpha(classification))


def expanded_alpha(state, num_outcomes=2):
    return state // num_outcomes


def expanded_log_zero_one(num_outcomes=2):
    r"""Log zero-one loss on the expanded state space.

    Returns the zero-one loss for the expanded state space

        >>> def loss(state, classification):
        >>>     return state//num_outcomes != classification//num_outcomes

    Use the returned function for computing losses.

    :param num_outcomes: Number of outcomes that dictates how the state space is
        expanded.
    :return: The loss function.
    """
    def alpha(s):
        return expanded_alpha(s, num_outcomes=num_outcomes)

    def loss(state, classification):
        return mapped_log_zero_one(state, classification, alpha)
    return loss
