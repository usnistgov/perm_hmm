"""This module contains the abstract class
:py:class:`~perm_hmm.policies.policy.PermPolicy`. This class provides
boilerplate to implement a policy for the permutation-based HMM.
"""

import warnings
import torch

from perm_hmm.util import flatten_batch_dims


class PermPolicy(object):
    """
    This is an abstract class that is used to select permutations. The get_perm
    method is called in-line when sampling with PermutedDiscreteHMM.  The
    get_perms method uses the get_perm method to compute all the permutations
    that would be chosen for a whole sequence.

    The boilerplate is to manage the shape of the incoming data, and to manage
    the history of the calculations used to compute the permutations. Because
    the calculation of permutations can be stateful, it is also useful to be
    able to "reset" the state of the calculation, which is done by the
    :py:meth:`~perm_hmm.policies.policy.PermPolicy.reset` method.

    Subclasses should implement the following methods:

    reset:
        Resets the state of the ``PermPolicy``. This method should be called
        after the ``PermPolicy`` is used to compute a sequence of
        permutations.  Subclasses should call the ``reset`` method of the parent
        class before cleaning up their own state.

    calculate_perm:
        Computes the permutation to apply, given the data observed. Should
        return both the permutation and the relevant parts of the calculation
        used to compute it, as a dictionary of tensors. If none are relevant,
        the return value should be the empty dictionary.

    .. seealso::
        :py:class:`~perm_hmm.policies.rotator_policy.RotatorPolicy`
        for an example of subclassing.

    The attributes of the ``PermPolicy`` are:

    ``possible_perms``:
        A tensor of shape ``(n_perms, n_states)``, that contains the allowable
        permutations.

    ``calc_history``:
        A list of dictionaries of tensors, containing the calculation history.

    ``perm_history``:
        A tensor of shape ``batch_shape + (n_steps, n_states)``, containing
        the permutations applied to the states thus far.
    """

    def __init__(self, possible_perms, save_history=False):
        """Initializes the PermPolicy.

        Should be called by the subclass constructor.

        :param possible_perms: The allowable permutations.
        :param save_history: Indicates whether to save the calculation
            and permutation history.
        """
        n_perms, n_states = possible_perms.shape
        if not (possible_perms.long().sort(-1).values ==
                torch.arange(n_states, dtype=torch.long).expand(
                    (n_perms, n_states)
                )).all():
            raise ValueError("The input permutations are not permutations of "
                             "the integers [0, ..., n_states]")
        self.possible_perms = possible_perms
        self.calc_history = {}
        self.perm_history = None
        self.save_history = save_history

    def _add_to_calc_history(self, calc_dict, shape):
        """Adds the step of the calculation to the calculation history.

        :param calc_dict: The dictionary containing the step of the calculation
            history.
        :return: None
        """
        for k, v in calc_dict.items():
            try:
                v = v.unsqueeze(1)
                v = v.reshape(shape + v.shape[1:])
            except (RuntimeError, ValueError, AttributeError, TypeError):
                if k in self.calc_history:
                    self.calc_history[k].append(v)
                else:
                    self.calc_history[k] = [v]
            else:
                if k in self.calc_history:
                    self.calc_history[k] = torch.cat(
                        (self.calc_history[k], v),
                        dim=len(shape),
                    )
                else:
                    self.calc_history[k] = v

    def _add_to_perm_history(self, perm, shape):
        """Adds the permutation used to the permutation history.

        :param perm: The permutation used.
        :return: None
        """
        perm = perm.unsqueeze(-2)
        perm = perm.reshape(shape + perm.shape[1:])
        if self.perm_history is None:
            self.perm_history = perm
        else:
            self.perm_history = torch.cat(
                (self.perm_history, perm),
                dim=-2,
            )

    def calculate_perm(self, data: torch.Tensor) -> (torch.Tensor, dict):
        """This method should be implemented by subclasses.

        Given the data, this method should return the permutation to apply,
        along with the relevant parts of the calculation used to compute it,
        in the form of a dictionary of tensors. If none are relevant, the second
        return value should be the empty dictionary.

        :param data: The data observed. The dimensions will always be
            ``(batch_len,) + event_shape``.
        :return: The permutation to apply, and the relevant parts of the
            calculation used to compute it, as a dictionary of tensors.
            The permutation should be a tensor of shape
            ``(batch_len, num_states)``, and the dictionary should contain
            tensors of shape ``(batch_len,) + arbitrary``.
        """
        raise NotImplementedError

    def get_perm(self, data: torch.Tensor, event_dims=0):
        """Takes an input of data from a single step, and returns a
        permutation.

        If self.save_history is True, the calculation and permutation history
        will be saved to the attributes self.calc_history and self.perm_history
        respectively.

        :param torch.Tensor data: Data from the HMM.
            shape ``batch_shape + event_shape``
        :param int event_dims: Number of event dimensions. Needed to distinguish
            the batch dimensions from the event dimensions. Should be equal to
            len(event_shape).
        :return: The permutation to be applied at the next step.
            shape ``batch_shape + (num_states,)``
        """
        data, shape = flatten_batch_dims(data, event_dims=event_dims)
        perm, calc_dict = self.calculate_perm(data)
        self._add_to_perm_history(perm, shape)
        if self.save_history:
            self._add_to_calc_history(calc_dict, shape)
        perm = perm.reshape(shape + (perm.shape[-1],))
        return perm

    def reset(self, save_history=False):
        """Resets the calculation history.

        Subclasses should call this method in their reset methods.

        :param save_history: Indicates whether to save the permutation and
            calculation histories the next time the policy is used to compute
            permutations.
        :return: None
        """
        self.perm_history = None
        self.save_history = save_history
        self.calc_history = {}

    def get_perms(self, data, event_dims=0):
        r"""
        Given a run of data, returns the permutations which would be applied.

        This should be used to precompute the permutations for a given model
        and given data sequence.

        :param torch.Tensor data: The sequence of data to compute the
            permutations for, of shape
            ``batch_shape + (time_len,) + event_shape``.
        :param int event_dims: Number of event dimensions. Needed to distinguish
            the batch dimensions from the event dimensions. Should be equal to
            len(event_shape).
        :returns: A tensor containing the permutations that would have been
            applied, given the input data, of shape
            ``batch_shape + (time_dim, num_states)``.
        """
        if self.perm_history:
            warnings.warn("The perm_history is not empty. "
                          "The returned perms will include these, "
                          "maybe you meant to call reset() before"
                          "calling this function?")
        shape = data.shape[:len(data.shape) - event_dims]
        max_t = shape[-1]
        for i in range(max_t):
            _ = self.get_perm(
                data[(..., i) + (slice(None),) * event_dims],
                event_dims=event_dims,
            )
        return self.perm_history
