import torch
import perm_hmm.loss_functions as lf
from perm_hmm.util import ZERO


def expanded_log_zero_one(state, classification):
    sl = state // 2
    cl = classification // 2
    loss = sl != cl
    floss = loss.float()
    floss[~loss] = ZERO
    log_loss = floss.log()
    log_loss[~loss] = 2*log_loss[~loss]
    return log_loss


def test_conditional_lzo():
    s = torch.tensor([
        [0, 4, 0, 3, 3, 3, 0],
        [0, 4, 0, 3, 3, 3, 1],
        [0, 4, 1, 3, 3, 2, 0],
    ], dtype=int)
    c = torch.tensor([
        [0, 4, 0, 2, 2, 3, 0],
        [0, 4, 0, 3, 3, 3, 0],
        [0, 4, 1, 3, 3, 2, 1],
    ], dtype=int)
    l = lf.expanded_log_zero_one(2)
    v = l(s, c)
    vp = expanded_log_zero_one(s, c)
    assert v.allclose(vp)
