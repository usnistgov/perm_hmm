"""Make a belief tree labelled with costs, then select the paths giving lowest
costs.

This module contains the
:py:class:`~perm_hmm.policies.min_tree.MinTreePolicy` class, which is a
:py:class:`~perm_hmm.policies.policy.PermPolicy` that selects the
permutations that minimize the cost, computed using a belief tree.
"""
import warnings
from copy import deepcopy
import numpy as np
import torch

from perm_hmm.policies.belief_tree import HMMBeliefTree
from perm_hmm.util import all_strings, perm_idxs_from_perms
from perm_hmm.models.hmms import random_phmm
from perm_hmm.policies.policy import PermPolicy
from perm_hmm.policies.belief import HMMBeliefState
import perm_hmm.log_cost as cf


class MinTreePolicy(PermPolicy):
    r"""Select the permutations that minimize the cost.

    With a limited look-ahead depth, we can compute the cost of all possible
    belief states, and then select the permutations that minimize the cost.

    In addition to the attributes of the base class, instances of this class
    have the following attributes:

    ``data_to_idx``:
        A function mapping data to indices.

    ``log_cost_func``:
        A function that computes the log-cost of a belief state.

    ``height``:
        The height of the belief tree.

    ``hmm``:
        The HMM used to compute the belief states.

    ``tree``:
        The tree of belief states.
    """

    def __init__(self, possible_perms, hmm, log_cost_func, look_ahead, data_to_idx=None, root_belief=None, initialize_tree=True, save_history=False):
        r"""Initialize the MinTreePolicy.

        This method does not initialize the tree. After instantiation, call the
        method :py:meth:`~perm_hmm.policies.min_tree.MinTreePolicy.initialize_tree`
        to initialize the tree.

        :param possible_perms: The allowable permutations.
        :param hmm: The HMM used to compute the belief states.
        :param log_cost_func: The function used to compute the log-cost of a
            belief state.
        :param int look_ahead: Number of steps to look ahead in tree calculation
        :param data_to_idx: The function to convert incoming data to indices
            corresponding to data. Defaults to ``lambda data: data.long()``.
        :param HMMBeliefState root_belief: The initial belief to seed the belief
            tree with.
        :param bool initialize_tree: Whether to call the initialize_tree method
            as a part of initialization of the object. If ``True``, will call
            initialize_tree() with no arguments. To specify arguments, pass
            ``False`` to this flag and call initialize_tree() separately.
        :param save_history: Indicates whether to save the history of the
            computation.
        """
        super().__init__(possible_perms, save_history=save_history)
        if data_to_idx is None:
            def data_to_idx(x):
                return x.long()
        self.data_to_idx = data_to_idx
        self.log_cost_func = log_cost_func
        self.hmm = deepcopy(hmm)
        self.look_ahead = look_ahead
        self.root_belief = root_belief
        if initialize_tree:
            self.initialize_tree()
        else:
            self.tree = None  # type: HMMBeliefTree | None

    def initialize_tree(self, look_ahead=None, root_belief=None, data_len=None):
        r"""Initializes the belief tree.

        Computes the belief tree, and saves it in the attribute :py:attr:`tree`.

        :param look_ahead: The number of steps to look ahead. The resulting
            tree has depth ``2*look_ahead + 1``
        :param root_belief: The root belief state. If None, defaults to the
            initial belief state of the HMM.
        :param data_len: The length of the data to be observed. If None,
            defaults to 1, so that the dimension will broadcast with later
            operations.
        :return: None
        """
        if look_ahead is not None:
            self.look_ahead = look_ahead
        if root_belief is not None:
            self.root_belief = root_belief
        self.tree = HMMBeliefTree(self.hmm, self.possible_perms, self.look_ahead, root_belief=self.root_belief, data_len=data_len)
        if self.save_history:
            if b'initial_tree' not in self.calc_history:
                self.calc_history[b'initial_tree'] = deepcopy(self.tree)
            else:
                warnings.warn('Initial tree already exists, so a new one will '
                              'not be saved. Did you remember to call reset()?')

    def reset(self, initialize_tree=True, save_history=False):
        r"""Resets the MinTreePolicy.

        This method resets the tree to None, and the history of the computation.

        :param initialize_tree: Whether to initialize the tree. If ``False``,
            tree is set to ``None``.
        :param save_history: Whether to save the history of the computation the
            next time the policy is used.
        :return: None
        """
        super().reset(save_history=save_history)
        if initialize_tree:
            self.initialize_tree()
        else:
            self.tree = None

    def calculate_perm(self, data: torch.Tensor) -> (torch.Tensor, dict):
        r"""Generates permutations that minimize the cost.

        This method updates the tree on-line. After receiving the data, it
        prunes the tree, expands the leaves, then recomputes the log-cost of the
        root and the corresponding permutations. It then returns the minimum
        cost permutation, and the corresponding cost.

        :param data: The data to compute the next permutation for.
        :return: The permutation that minimizes the cost, and a dict containing
            keys

                ``b'log_costs'``: The tree of costs at each stage.

                ``b'perm_idx_tree'``: The index of the cost minimizing
                    permtuation. This is a tree of

                    .. math::

                        \operatorname{argmin}_{\sigma^n}c\left( Y_k^n | \sigma^n y^{k-1} \right)

                    for a cost function :math:`c`, computed from

                    .. math::
                        \operatorname{min}_{\sigma^n}\mathbb{E}_{Y^n_k|\sigma^n, y^{k-1}}
                        \left[c\left( Y^n_k | \sigma^n, y^{k-1} \right)\right]
                        = \min_{\sigma_k}\mathbb{E}_{Y_k|\sigma^k, y^{k-1}}\left[ \cdots
                        \min_{\sigma_{n-1}}\mathbb{E}_{Y_{n-1}|\sigma^{n-1}, y^{n-2}}
                        \left[\min_{\sigma_n}\mathbb{E}_{Y_n|\sigma^n, y^{n-1}}
                            \left[
                                c\left( Y^n_k | \sigma^n y^{k-1} \right)
                        \right]\right] \cdots\right]

                    That is, these are the indices of the permutations that
                    minimize the expected cost, if the process were to terminate
                    in ``depth`` steps.

                ``b'penultimate_layer'``: These are the newly computed beliefs
                    obtained from transitioning with each possible permutation.
                    Has dimensions -1: s_k, -2: s_0, 0: batch. All other
                    dimensions are for the "tree" dimensions. That is, for the
                    ``i``th run, for a sequence of permutation indices
                    ``sigma[0]`` to ``sigma[depth-1]`` and a sequence of
                    observation indices ``o[0]`` to ``o[depth-2]``, the belief
                    state :math:`p(s_0, s_k|y^k \sigma^k)` that would have been
                    obtained is given by::

                        calc_dict[b'penultimate_layer'][i, o[0], sigma[0], ...,
                            o[depth-2], sigma[depth-1], s_0,  s_k]

                    This is really the "penultimate layer" of the tree of belief
                    states that is recomputed.

                    Instead of the whole belief tree at each stage, This along
                    with ``b'leaves'`` is returned. From the returned
                    information, one should be able to reconstruct the whole
                    belief tree at each stage.

                ``b'leaves'``: This is the newly computed final layer of the
                    belief tree. The cost function is computed on this layer,
                    then passed up the tree to compute the permutations.
        """
        if not self.tree:
            raise ValueError("Must call initialize_tree() first.")
        tree_len = self.tree.beliefs[0].logits.shape[-3]
        if data.shape[0] != tree_len:
            if tree_len == 1:
                self.tree.broadcast_to_length(data.shape[0])
            else:
                raise ValueError("Must call reset() first.")
        self.tree.prune_tree(self.data_to_idx(data))
        self.tree.grow()
        perm_idx_tree, log_costs = self.tree.perm_idxs_from_log_cost(self.log_cost_func, return_log_costs=True)
        perm_idx = perm_idx_tree.perm_idxs[0]
        perm = self.possible_perms[perm_idx]
        self.tree.prune_tree(perm_idx)
        return perm, {
            b'log_costs': log_costs,
            b'perm_idx_tree': perm_idx_tree,
            b'penultimates': torch.tensor(np.rollaxis(self.tree.beliefs[-2].logits.numpy(), -3)),
            b'leaves': torch.tensor(np.rollaxis(self.tree.beliefs[-1].logits.numpy(), -3))
        }

    def penultimates_from_sequence(self, data, perms, event_dims=0):
        r"""From a sequence of data and corresponding permutations, computes the
            belief trees that would have been obtained, had those choices been
            made.

        :param data: The hypothetical data to compute beliefs for.
        :param perms: The hypothetical permtuations to compute beliefs for.
        :param event_dims: The number of dimensions of the data that correspond
            to event dimensions.
        :return: A list of trees.
        :raises ValueError: If tree is not initialized.
        """
        fixed = TreeFixedPolicy(self.possible_perms, self.hmm, perms, self.look_ahead, self.data_to_idx, root_belief=self.root_belief, save_history=True)
        _ = fixed.get_perms(data, event_dims=event_dims)
        d = fixed.calc_history
        return d[b'penultimates']


class TreeFixedPolicy(MinTreePolicy):

    def __init__(self, possible_perms, hmm, perms, look_ahead, data_to_idx=None, root_belief=None, initialize_tree=True, save_history=False):
        r"""Initializes the policy.

        Needs the ``perms`` argument, that is the permutations that will be
        returned, independent of the data.

        :param torch.Tensor possible_perms: The allowed permutations.
        :param perm_hmm.models.hmms.PermutedDiscreteHMM hmm:
            The HMM used to calculate the belief states.
        :param torch.Tensor perms: The fixed sequence of permutations to be returned.
        :param int look_ahead: The number of steps to look ahead when computing
            the belief tree. The belief tree will have height 2*look_ahead+1.
        :param data_to_idx: The function to convert incoming data to indices
            corresponding to data. Defaults to ``lambda data: data.long()``.
        :param HMMBeliefState root_belief: The belief state to seed the belief
            tree with. Defaults to the initial state of the HMM.
        :param bool initialize_tree: Whether to call the initialize_tree method
            as a part of initialization of the object. If ``True``, will call
            initialize_tree() with no arguments. To specify arguments, pass
            ``False`` to this flag and call initialize_tree() separately.
        :param bool save_history: Whether to save the computation history.
        """
        super().__init__(
            possible_perms,
            hmm,
            lambda x: None,
            look_ahead,
            data_to_idx=data_to_idx,
            root_belief=root_belief,
            initialize_tree=initialize_tree,
            save_history=save_history
        )
        self.step = 0
        self.perms = perms
        self.perm_idxs = perm_idxs_from_perms(self.possible_perms, self.perms)

    def reset(self, initialize_tree=True, save_history=False):
        r"""Resets the policy.

        Resets the step to 0, and the computation history.

        :param bool initialize_tree: Whether to initialize the belief tree.
        :param bool save_history: Whether to save the computation history.
        :return: None
        """
        super().reset(
            initialize_tree=initialize_tree,
            save_history=save_history
        )
        self.step = 0

    def calculate_perm(self, data: torch.Tensor) -> (torch.Tensor, dict):
        if not self.tree:
            raise ValueError("Must call initialize_tree() first.")
        self.tree.prune_tree(self.data_to_idx(data))
        self.tree.grow()
        perm_idx = self.perm_idxs[self.step]
        perm_idx = perm_idx.expand(data.shape[:1] + perm_idx.shape)
        perm = self.perms[self.step, :]
        perm = perm.expand(data.shape[:1] + perm.shape)
        self.tree.prune_tree(perm_idx)
        self.step += 1
        return perm, {
            b'penultimates': torch.tensor(np.rollaxis(self.tree.beliefs[-2].logits.numpy(), -3)),
            b'leaves': torch.tensor(np.rollaxis(self.tree.beliefs[-1].logits.numpy(), -3))
        }


def log_entropy_of_expanded(logits, n_outcomes):
    r"""Given a joint distribution
    :math:`\mathbb{P}((l_0, y_0), (l_k, y_k)|y^{k-1})`, computes the log of the
    entropy of the initial physical state, :math:`\log H(L_0|y^{k-1})`.

    When we allow the next "physical" state to also depend on the previous
    outcome, we account for this by using an expanded state space that is the
    cartesian product of the physical states and the possible outcomes.
    Belief states over such an expanded state space are then joint distributions
    :math:`\mathbb{P}((l_0, y_0), (l_k, y_k)|y^{k-1})`, where :math:`l_k` is
    the next physical state, and :math:`l_0` is the initial physical state.

    This function computes the log of the entropy of the initial physical state
    for such a joint distribution.

    :param logits: :math:`\mathbb{P}((l_0, y_0), (l_k, y_k)|y^{k-1})`, with
        dimensions -1: (l_k, y_k), -2: (l_0, y_0).
    :param n_outcomes: The number of possible outcomes. Uses this to reshape the
        joint distribution, then marginalizes over the outcomes.
    :return: The log of the entropy of the initial physical state.
    """
    logits = logits.reshape(
        logits.shape[:-2] + (-1, n_outcomes) + logits.shape[-1:])
    logits = logits.logsumexp(-2)
    return cf.log_initial_entropy(logits)


class ExpandedEntPolicy(MinTreePolicy):
    r"""A policy that uses the conditional entropy of the initial physical
    state as the cost function.

    When we allow the next "physical" state to also depend on the previous
    outcome, we account for this by using an expanded state space that is the
    cartesian product of the physical states and the possible outcomes.

    In this model, the initial state has no observation so that the belief state
    must be initialized differently from
    :py:class:`~perm_hmm.policies.min_ent.MinEntropyPolicy`, and the cost
    function must be computed differently as well.

    .. seealso:: :py:class:`~perm_hmm.policies.min_ent.MinEntropyPolicy`
    """

    def __init__(self, possible_perms, hmm, look_ahead=1,
                 data_to_idx=None, trivial_obs=None, root_belief=None,
                 initialize_tree=True, save_history=False):
        r"""Initializes the policy.

        :param possible_perms: The allowable permutations.
        :param hmm: The HMM used to compute likelihoods.
        :param look_ahead: The number of steps to look ahead.
        :param data_to_idx: The function to convert data to indices.
        :param trivial_obs: The observation :math:`y_0` such that
            :math:`\mathbb{P}((l_0, y_0')) = \delta_{y_0, y_0'}\mathbb{P}(l_0)`.
        :param HMMBeliefState root_belief: The initial belief to seed the belief
            tree with. Defaults to the state

            .. math::

                p(s_0, s_k) = A(s_k|s_0)\pi(s_0)

            where :math:`\pi` is the initial distribution of the HMM and
            :math:`A` is the transition matrix of the HMM. This is done because
            in the expanded state space model, the initial state does not emit
            an observation.
        :param bool initialize_tree: Whether to call the initialize_tree method
            as a part of initialization of the object. If ``True``, will call
            initialize_tree() with no arguments. To specify arguments, pass
            ``False`` to this flag and call initialize_tree() separately.
        :param save_history: Indicates whether to save the history of the
            calculation.
        """
        def _expanded_entropy(logits):
            return log_entropy_of_expanded(
                logits,
                n_outcomes=hmm.enumerate_support(expand=False).squeeze(-1).shape[-1]
            )
        self.trivial_obs = trivial_obs
        if root_belief is None:
            root_belief = HMMBeliefState.from_skipfirsthmm(
                hmm,
                trivial_obs=self.trivial_obs
            )
        super().__init__(
            possible_perms,
            hmm,
            _expanded_entropy,
            look_ahead,
            root_belief=root_belief,
            initialize_tree=initialize_tree,
            data_to_idx=data_to_idx,
            save_history=save_history
        )


class MinEntPolicy(MinTreePolicy):

    def __init__(self, possible_perms, hmm, look_ahead=1, data_to_idx=None, root_belief=None, initialize_tree=True, save_history=False):
        log_cost_func = cf.log_initial_entropy
        super().__init__(
            possible_perms,
            hmm,
            log_cost_func,
            look_ahead,
            root_belief=root_belief,
            initialize_tree=initialize_tree,
            data_to_idx=data_to_idx,
            save_history=save_history,
        )


def main():
    nstates = 2
    nsteps = 3
    hmm = random_phmm(nstates)
    possible_perms = torch.eye(nstates, dtype=torch.long)
    policy = MinTreePolicy(possible_perms, hmm,
                           lambda x: cf.log_renyi_entropy(x, 2.0), 2)
    data = all_strings(nsteps)
    perms = policy.get_perms(data)
    print(perms)


if __name__ == '__main__':
    main()
