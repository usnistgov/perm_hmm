r"""Exhaustively searches the best permutations for all possible observations.

This is a policy that exhaustively searches the best permutations for all
possible observations. This method is very slow and should only be used for
testing purposes. The complexity of this method is O((n*p)**t), where n is the
number of possible observations, p is the number of possible permutations, and
t is the number of steps. The computation works by generating a tree of all
possible observations and permutations, and then using the `Bellman equation`_
to compute the best permutation for each observation.

.. _`Bellman equation`: https://en.wikipedia.org/wiki/Bellman_equation
"""
import os
from copy import deepcopy
import torch
from perm_hmm.util import id_and_transpositions, all_strings
from perm_hmm.policies.belief import HMMBeliefState
from perm_hmm.policies.belief_tree import HMMBeliefTree, PermIdxTree
from perm_hmm.models.hmms import random_phmm
from perm_hmm.policies.policy import PermPolicy
import perm_hmm.log_cost as cf


class ExhaustivePolicy(PermPolicy):
    r"""Exhaustively searches the best permutations for all possible
    observations.

    This is a policy that exhaustively searches the best permutations for all
    possible observations. This method is very slow and should only be used for
    testing purposes. The complexity of this method is O((n*p)**t), where n is
    the number of possible observations, p is the number of possible
    permutations, and t is the number of steps.

    Instances of this class have the following attributes:

    data_to_idx:
        A function that maps the data to indices. If unspecified in
        initialization, defaults to::

            def data_to_idx(x):
                return x.long()

    look_ahead:
        :py:class:`~int` that indicates the number of steps to compute for.

    root_belief:
        The belief state that the tree of belief states is rooted at.

    hmm:
        :py:class:`~perm_hmm.models.hmms.PermutedDiscreteHMM` that is used to
        compute belief states.

    belief_tree:
        List of :py:class:`~perm_hmm.policies.belief.HMMBeliefState`s. Each
        element of the list corresponds to a layer of the tree.
        Computed using the method
        :py:meth:`~perm_hmm.policies.policy.PermPolicy.initialize_tree`.

    perm_tree:
        List of tensors. Gives the best permutation for each observation.
        Computed using the method
        :py:meth:`~perm_hmm.policies.exhaustive.ExhaustivePolicy.compute_perm_tree`.

    remaining_tree:
        When actually using the policy, this is the tree that is used to
        generate permutations. This tree is reset when using the method
        :py:meth:`~perm_hmm.policies.exhaustive.ExhaustivePolicy.reset`.
    """

    def __init__(self, possible_perms, hmm, look_ahead, data_to_idx=None, initialize_tree=True, root_belief=None, terminal_offset=False, save_history=False):
        r"""Initializes the policy.

        :param possible_perms: The possible permutations to select from at each
            step.
        :param hmm: The HMM to compute likelihoods with.
        :param data_to_idx: The mapping from data to indices.
        :param save_history: Whether to save the calculation history for the
            policy.
        """
        super().__init__(possible_perms, save_history=save_history)
        if data_to_idx is None:
            def data_to_idx(x):
                return x.long()
        self.data_to_idx = data_to_idx
        self.hmm = deepcopy(hmm)
        self.look_ahead = look_ahead
        self.root_belief = root_belief
        if initialize_tree:
            self.initialize_tree(self.look_ahead, root_belief=root_belief, terminal_offset=terminal_offset)
        else:
            self.belief_tree = None  # type: HMMBeliefTree | None
        self.perm_tree = None  # type: None | PermIdxTree
        self.remaining_tree = None  # type: None | PermIdxTree

    def initialize_tree(self, num_steps, root_belief=None, terminal_offset=False):
        r"""Computes the full tree of beliefs.

        This method computes the full tree of beliefs, which is used to select
        the best permutations. This method is very expensive, using memory
        O(n*(n*p)**t), where n is the number of possible observations, p is the
        number of possible permutations, and t is the number of steps.

        :param int num_steps: Number of steps to compute for.
        :param torch.Tensor root_belief: The initial belief state. Defaults to
            the initial distribution of the HMM.
        :param bool terminal_offset: Indicates whether the leaves of the tree
            should have offset=True, i.e. should be indexed by observations.
        :return: None
        """
        self.belief_tree = HMMBeliefTree(self.hmm, self.possible_perms, num_steps, root_belief=root_belief, terminal_offset=terminal_offset)

    def compute_perm_tree(self, log_cost=None, return_log_costs=False, delete_belief_tree=True, terminal_log_cost=None, is_cost_func=True):
        r"""After computing the full tree of beliefs using :py:meth:`~perm_hmm.policies.exhaustive.ExhaustivePolicy.initialize_tree`,
        this method is used to compute the best permutations for each
        observation.

        :param log_cost: A function for the cost of a terminal belief state.
            Defaults to the log of the min entropy of the initial state.
        :param bool return_log_costs: Indicates whether to return the optimal costs
            of the permutations.
        :param bool delete_belief_tree: Indicates whether to delete the belief
            tree after computing the permutations.
        :param terminal_log_cost: The log cost attached to the terminal belief
            states. If unspecified, will compute the log cost using the function
            log_cost.
        :return: If return_log_costs is true, returns the log costs of the
            optimal permutations at all nodes of the tree. Otherwise, returns
            None.
        :raises: ValueError if the belief tree has not been computed.
        """
        if self.belief_tree is None:
            raise Exception('Must compute belief tree first.')
        if log_cost is None:
            log_cost = cf.min_entropy
            is_cost_func = False
        r = self.belief_tree.perm_idxs_from_log_cost(log_cost, return_log_costs=return_log_costs, terminal_log_cost=terminal_log_cost, is_cost_func=is_cost_func)
        if return_log_costs:
            self.perm_tree, log_costs = r
        else:
            self.perm_tree = r
        self.perm_tree.trim_list_tree()
        if delete_belief_tree:
            del self.belief_tree
            self.belief_tree = None
        if return_log_costs:
            return log_costs

    def reset(self, save_history=False):
        r"""Resets the policy.

        Because computing the full tree of beliefs is expensive, this method
        only resets the tree ``remaining_tree``, which acts as a cache.

        :param bool save_history: Indicates whether to save the calculation
            history for the policy.
        :return: None
        """
        super().reset(save_history=save_history)
        self.remaining_tree = None

    def calculate_perm(self, data):
        r"""Generates the best permutation for the given data.

        This method is called after using the
        :py:meth:`~perm_hmm.policies.exhaustive.ExhaustivePolicy.initialize_tree`
        and
        :py:meth:`~perm_hmm.policies.exhaustive.ExhaustivePolicy.compute_perm_tree`
        methods.

        HACK: The last perm to be returned will be some arbitrary permutation.
        This shouldn't matter, as the last permutation acts after the last data.

        :param data: The data observed.
        :return: The best permutation to apply having seen that data.
        """
        data_len = data.shape[0]
        if self.remaining_tree is None:
            if self.perm_tree is None:
                raise ValueError("Must compute perm tree first. Call "
                                 "compute_perm_tree.")
            self.remaining_tree = deepcopy(self.perm_tree)
            self.remaining_tree.expand_batch(data_len)
        # HACK: Last perm to be returned acts after the last data. Just return
        # Perm index 0 as default.
        if len(self.remaining_tree.perm_idxs) == 0:
            perm_idx = torch.zeros(data_len, dtype=torch.long)
        else:
            data_idx = self.data_to_idx(data)
            perm_idx = self.remaining_tree.perm_idxs[0][data_idx, torch.arange(data_len)]
        perm = self.possible_perms[perm_idx]
        if len(self.remaining_tree.perm_idxs) != 0:
            self.remaining_tree.prune_perm_tree(data_idx)
        return perm, {}


def subtree_cost(logits, hmm, possible_perms, num_steps, filename=None):
    policy = ExhaustivePolicy(possible_perms, hmm, num_steps, root_belief=HMMBeliefState(logits, hmm, offset=True))
    cost = policy.compute_perm_tree(return_log_costs=True, delete_belief_tree=False)
    d = {b'log_values': cost, b'beliefs': policy.belief_tree, b'perms': policy.perm_tree.perm_idxs}
    if filename is not None:
        with open(filename, 'wb') as f:
            torch.save(d, f)
    return cost[0]


def split_path(path):
    return [x for i, x in enumerate(path) if i % 2 == 0], [x for i, x in enumerate(path) if i % 2 == 1]


def name_from_path(path):
    obs, perm = split_path(path)
    return 'obs' + '_'.join(map(str, obs)) + '_perm_' + '_'.join(map(str, perm)) + '.pt'


def _all_cost_helper(leaf_logits, hmm, possible_perms, subtree_steps, path_to_root, directory=None, save=False):
    if len(leaf_logits.shape) == 3:
        if save:
            filename = name_from_path(path_to_root)
            if directory is not None:
                filename = os.path.join(directory, name_from_path(path_to_root))
        else:
            filename = None
        return subtree_cost(leaf_logits, hmm, possible_perms, subtree_steps, filename=filename)
    else:
        return torch.stack([_all_cost_helper(l, hmm, possible_perms, subtree_steps, path_to_root + [i], directory, save=save) for i, l in enumerate(leaf_logits)])


def all_subtree_costs(belief_leaves, hmm, possible_perms, subtree_steps, directory=None, save=False):
    costs = _all_cost_helper(belief_leaves.logits, hmm, possible_perms, subtree_steps, [], directory=directory, save=save)
    return costs


class SplitExhaustivePolicy(ExhaustivePolicy):

    def __init__(self, possible_perms, hmm, look_ahead, split=None, data_to_idx=None, root_belief=None, save_history=False):
        if split is None:
            split = look_ahead - look_ahead // 2
        super().__init__(
            possible_perms,
            hmm,
            split,
            data_to_idx=data_to_idx,
            initialize_tree=True,
            root_belief=root_belief,
            terminal_offset=True,
            save_history=save_history,
        )
        self.total_look_ahead = look_ahead
        self.initial_cost = None

    def make_initial_cost(self, root_directory=None, save=False):
        self.initial_cost = all_subtree_costs(self.belief_tree.beliefs[-1], self.hmm, self.possible_perms, self.total_look_ahead - self.look_ahead + 1, directory=root_directory, save=save)

    def compute_perm_tree(self, log_cost=None, return_log_costs=True,
                          delete_belief_tree=False, terminal_log_cost=None,
                          is_cost_func=False):
        if terminal_log_cost is None:
            if self.initial_cost is None:
                raise ValueError("Please call make_initial_cost first.")
            terminal_log_cost = self.initial_cost
        return super().compute_perm_tree(
            return_log_costs=return_log_costs,
            terminal_log_cost=terminal_log_cost,
            delete_belief_tree=delete_belief_tree,
            is_cost_func=is_cost_func,
        )


def main():
    nstates = 3
    nsteps = 4
    hmm = random_phmm(nstates)
    possible_perms = id_and_transpositions(nstates)
    policy = ExhaustivePolicy(possible_perms, hmm, nsteps)
    v = policy.compute_perm_tree(return_log_costs=True)
    all_data = all_strings(nsteps)
    p = policy.get_perms(all_data)
    print(p)
    print(v[0])


if __name__ == '__main__':
    main()
