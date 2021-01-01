import torch

from perm_hmm.util import ZERO


def log_zero_one(state, classification):
    loss = classification != state
    floss = loss.float()
    floss[~loss] = ZERO
    log_loss = floss.log()
    log_loss[~loss] = 2*log_loss[~loss]
    return log_loss


def zero_one(state, classification):
    return classification != state


def binary_zero_one(dark_state, bright_state):
    testing_states = torch.tensor([dark_state, bright_state], dtype=int)

    def _wrapper(state, classification):
        return zero_one(state, testing_states[classification])
    return _wrapper


def log_binary_zero_one(dark_state, bright_state):
    testing_states = torch.tensor([dark_state, bright_state], dtype=int)

    def _wrapper(state, classification):
        return log_zero_one(state, testing_states[classification])
    return _wrapper
