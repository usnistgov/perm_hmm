import torch


class PermSelector(object):

    def __init__(self, possible_perms):
        n_perms, state_dim = possible_perms.shape
        if not (possible_perms.long().sort(-1).values ==
                torch.arange(self.state_dim, dtype=torch.long).expand(
                    (n_perms, self.state_dim)
                )).all():
            raise ValueError("The input permutations are not permutations of "
                             "the integers [0, ..., state_dim]")
        self.possible_perms = possible_perms
        self.history = {}

    def perm(self, data: torch.Tensor, save_history=False):
        raise NotImplementedError

    def start(self, shape, save_history=False):
        raise NotImplementedError
