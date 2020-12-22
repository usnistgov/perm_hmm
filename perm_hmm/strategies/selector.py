import torch
from operator import mul
from functools import reduce
from functools import wraps
from perm_hmm.return_types import PermWithHistory




class PermSelector(object):
    """
    A description of what an algorithm which selects permutations
    should do, at a minimum.

    A real algorithm should include a model, and presumably
    a calibration method.
    """

    def __init__(self, possible_perms, save_history=False):
        n_perms, n_states = possible_perms.shape
        if not (possible_perms.long().sort(-1).values ==
                torch.arange(n_states, dtype=torch.long).expand(
                    (n_perms, n_states)
                )).all():
            raise ValueError("The input permutations are not permutations of "
                             "the integers [0, ..., n_states]")
        self.possible_perms = possible_perms
        self._calc_history = {}
        self._perm_history = []
        self.shape = None
        self.save_history = save_history

    @classmethod
    def manage_shape(cls, perm):
        @wraps(perm)
        def _wrapper(self, *args, **kwargs):
            event_dims = kwargs.get("event_dims", 0)
            try:
                data_shape = args[0].shape
                shape = data_shape[:len(data_shape) - event_dims]
            except (AttributeError, IndexError):
                shape = None
            self_shape = getattr(self, "shape", None)
            if self_shape is None:
                if shape is not None:
                    self.shape = shape
            data = args[0]
            if shape is not None:
                data = data.reshape((reduce(mul, self.shape, 1),) + data_shape[len(data_shape) - event_dims:])
            perms = perm(self, data, *args[1:], **kwargs)
            if shape is not None:
                perms.reshape(shape + perms.shape[-1:])
            return perms
        return _wrapper

    @classmethod
    def manage_calc_history(cls, perm):
        @wraps(perm)
        def _wrapper(self, *args, **kwargs):
            save_history = getattr(self, "save_history", False)
            retval = perm(self, *args, **kwargs)
            perms, calc_history = retval
            if save_history:
                for k, v in calc_history.items():
                    try:
                        self._calc_history[k].append(v)
                    except KeyError:
                        self._calc_history[k] = [v]
            return perms
        return _wrapper

    @classmethod
    def manage_perm_history(cls, perm):
        @wraps(perm)
        def _wrapper(self, *args, **kwargs):
            perms = perm(self, *args, **kwargs)
            self._perm_history.append(perms)
            return perms
        return _wrapper

    @property
    def perm_history(self):
        if len(self._perm_history) == 0:
            return torch.Tensor()
        else:
            try:
                toret = torch.stack(self._perm_history, dim=-2)
            except RuntimeError:
                return self._perm_history
            if len(self.shape) == 0:
                toret.squeeze_(0)
            return toret

    @perm_history.setter
    def perm_history(self, val):
        self._perm_history = val

    @perm_history.deleter
    def perm_history(self):
        del self._perm_history

    @property
    def calc_history(self):
        if len(self._calc_history) == 0:
            return self._calc_history
        if any([len(v) == 0 for v in self._calc_history.values()]):
            return self._calc_history
        if self.shape is None:
            return self._calc_history
        try:
            return {k: torch.stack([x.reshape(self.shape + x.shape[1:]) for x in v], dim=-v[0].ndim) for k, v in self._calc_history.items()}
        except RuntimeError:
            return self._calc_history

    @calc_history.setter
    def calc_history(self, val):
        self._calc_history = val

    @calc_history.deleter
    def calc_history(self):
        del self._calc_history

    def perm(self, data: torch.Tensor, shape=()):
        """
        Takes a (vectorized) input of data from a single time step,
        and returns a (correspondingly shaped) permutation.
        :param torch.Tensor data: Data from the HMM.
            shape ``sample_shape + batch_shape + hmm.observation_dist.event_shape``
        :param save_history: A flag indicating whether or not to
            save the history of the computation involved to produce the
            permutations. The function shouldn't return anything
            different even if this flag is true, but the history should be
            available in the .history attribute at the end of the run.
        :return: The permutation to be applied at the next time step.
            shape ``(n_batches, n_states)``
        """
        raise NotImplementedError

    def reset(self, save_history=False):
        pass

    def get_perms(self, data, time_dim, save_history=False):
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
        m = len(d_shape)
        if time_dim < 0:
            obs_event_dim = -(time_dim + 1)
        else:
            obs_event_dim = m - (time_dim + 1)
        shape = d_shape[:m - obs_event_dim]
        max_t = shape[-1]
        perms = []
        self.reset(save_history=save_history)
        for i in range(max_t):
            perms.append(self.perm(
                data[(..., i) + (
                    slice(None),) * obs_event_dim],
            ))
        perms = torch.stack(perms, -2)
        return perms
