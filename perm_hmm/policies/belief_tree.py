r"""Provides functions used by strategies that use a tree to select the
permutation.

To compute optimal permutations, we use the belief states
.. math::

    b(y^{k-1}) := \mathbb{P}(s_0, s_k|y^{k-1}),

where the :math:`s_k` are the states of the HMM at step :math:`k`, and the
superscript :math:`y^{k-1}` is the sequence of observations up to step
:math:`k-1`.

Here, when we refer to a tree, we really mean a list of
:py:class:`~perm_hmm.strategies.belief.HMMBeliefState` objects. The i'th
object contains the beliefs for all the nodes at the i'th level of the tree.
"""
import torch

from perm_hmm.policies.belief import HMMBeliefState


class HMMBeliefTree(object):
    r"""
    Instances of this class have the following attributes:

    ``hmm``:
        A :py:class:`~perm_hmm.models.hmms.PermutedDiscreteHMM` that is used to
        calculate belief states.

    ``possible_perms``:
        A :py:class:`~torch.Tensor` of type ``long`` that contains the possible
        permutations. This is used to compute transition matrices for updating
        belief states.
    """

    def __init__(self, hmm, possible_perms, nsteps, root_belief: HMMBeliefState = None, data_len=None, terminal_offset=False):
        r"""Generates the belief tree for the given HMM.

        Builds a tree that is traversed by sequences :math:`y_0, \sigma_0, y_1,
        \sigma_1, \ldots`, where the :math:`\sigma_k` are permutation indices, and
        the :math:`y_k` are the observation indices.  This tree has a layered
        structure. Attached to each node in the tree is a belief state
        :math:`\mathbb{P}(s_0, s_k|y^{k-1})`, or :math:`\mathbb{P}(s_0, s_k|y^k)`,
        depending on whether the node is an even or odd number of steps from the
        root, respectively. To go from a belief state attached to one node to a
        belief state attached to one of that node's children, we either use a
        transition or a Bayesian update, depending on whether the edge is a
        permutation or an observation, respectively.

        :param hmm: The HMM to compute likelihoods with.
        :param possible_perms: The allowable permutations.
        :param nsteps: The number of steps to compute for. (2 * nsteps + 1) is the
            height of the tree.
        :param HMMBeliefState root_belief: The belief state to start the tree with. If None,
            defaults to the initial state distribution of the HMM.
        :param data_len: The length of the data. If None, defaults to 1.
        :param terminal_offset: Whether the leaves of the tree should be labeled by
            observation indices.
        :return: A list of belief states, to be interpreted as a tree by looking at
            the ith element of the list as the set of all nodes at the ith level.
        """
        self.hmm = hmm
        self.possible_perms = possible_perms
        self._build_tree(nsteps, root_belief, data_len, terminal_offset)

    def _build_tree(self, nsteps, root_belief: HMMBeliefState = None, data_len=None, terminal_offset=False):
        r"""Generates the belief tree for the given HMM.

        Builds a tree that is traversed by sequences :math:`y_0, \sigma_0, y_1,
        \sigma_1, \ldots`, where the :math:`\sigma_k` are permutation indices, and
        the :math:`y_k` are the observation indices.  This tree has a layered
        structure. Attached to each node in the tree is a belief state
        :math:`\mathbb{P}(s_0, s_k|y^{k-1})`, or :math:`\mathbb{P}(s_0, s_k|y^k)`,
        depending on whether the node is an even or odd number of steps from the
        root, respectively. To go from a belief state attached to one node to a
        belief state attached to one of that node's children, we either use a
        transition or a Bayesian update, depending on whether the edge is a
        permutation or an observation, respectively.

        :param nsteps: The number of steps to compute for. (2 * nsteps + 1) is the
            height of the tree.
        :param root_belief: The belief state to start the tree with. If None,
            defaults to the initial state distribution of the HMM.
        :param data_len: The length of the data. If None, defaults to 1.
        :param terminal_offset: Whether the leaves of the tree should be labeled by
            observation indices.
        :return: A list of belief states, to be interpreted as a tree by looking at
            the ith element of the list as the set of all nodes at the ith level.
        :raise ValueError: If ``nsteps`` is less than 1. Must look ahead at
            least one step.
        """
        if nsteps < 1:
            raise ValueError("Cannot build a tree of less than 1 look ahead "
                             "steps.")
        if data_len is None:
            data_len = 1
        if root_belief is None:
            root_belief = HMMBeliefState.from_hmm(self.hmm)
            root_belief.logits = root_belief.logits.expand(data_len, -1, -1)
        self.beliefs = [root_belief]
        if terminal_offset and (nsteps == 1):
            return
        b = root_belief.bayes_update(self.hmm.observation_dist.enumerate_support(expand=False).squeeze(-1), new_dim=True)
        self.beliefs.append(b)
        if (not terminal_offset) and (nsteps == 1):
            return
        while len(self.beliefs) < (2 * (nsteps - 1)):
            self.grow(self.possible_perms)
        if not terminal_offset:
            self.grow(self.possible_perms)
        else:
            self.beliefs.append(self.beliefs[-1].transition(self.possible_perms, new_dim=True))

    def broadcast_to_length(self, length):
        new_beliefs = []
        for b in self.beliefs:
            shape = torch.broadcast_shapes((length, 1, 1), b.logits.shape)
            new_b = HMMBeliefState(b.logits.expand(shape).clone(), b.hmm, offset=b.offset)
            new_beliefs.append(new_b)
        self.beliefs = new_beliefs

    def grow(self, possible_perms=None, hmm=None):
        """Expands the tree by two levels.

        Assumes that the leaves have offset=True. Then, we expand the leaves by
        transitioning the belief states at the leaves, and then again by Bayesian
        updates.

        :param possible_perms: The allowable permutations.
        :param hmm: The HMM to compute likelihoods with.
        :return: An expanded tree, in the form of a list of belief states.
        """
        if possible_perms is None:
            possible_perms = self.possible_perms
        if hmm is None:
            hmm = self.hmm
        b = self.beliefs[-1].transition(possible_perms, hmm=hmm, new_dim=True)
        self.beliefs.append(b)
        b = self.beliefs[-1].bayes_update(hmm.observation_dist.enumerate_support(expand=False).squeeze(-1), hmm=hmm, new_dim=True)
        self.beliefs.append(b)

    def perm_idxs_from_log_cost(self, log_cost_func, return_log_costs=False, terminal_log_cost=None, is_cost_func=True):
        r"""Computes :math:`\mathbb{E}_{Y_k^n|y^{k-1}}[c(y^{k-1},Y_k^n)]` and the
        corresponding permutation indices that minimize this expectation.

        Given a tree of belief states, computes the expected cost of the tree.
        This computation is performed by first evaluating the cost function at the
        leaves of the tree, then propagating the cost up the tree.

        To compute the cost at an internal node whose children are labeled by data,
        we take the expectation over the children's costs, using the belief state
        to compute said expectation. To compute the cost at an internal node whose
        children are labeled by permutations, we take the minimum over the
        children's costs. This is a direct computation of the expected cost using
        the `Bellman equation`_.

        We then return both the permutation indices and, if ``return_costs`` is
        True, the expected cost.

        The computation is done in log space, so the cost function must be in log
        space as well.

        .. _`Bellman equation`: https://en.wikipedia.org/wiki/Bellman_equation

        :param log_cost_func: The cost function to compute the expected cost of.
            Must be in log space, and must take a single argument, which is a
            tensor of shape ``tree_shape + (n_states, n_states)``, returning a
            tensor of shape ``tree_shape``. The last two dimensions of the input
            correspond to the initial and final states of the HMM.
        :param bool return_log_costs: Whether to return the expected cost as well.
        :param terminal_log_cost: A tensor of terminal costs to start the calculation
            with. Defaults to ``log_cost_func(self.tree[-1].logits)``
        :return: A list of permutation indices, and, if ``return_costs`` is True,
            the expected cost.
        """
        if terminal_log_cost is None:
            terminal_log_cost = log_cost_func(self.beliefs[-1].logits)
        costs = [terminal_log_cost]
        perm_idxs = []
        for b in reversed(self.beliefs[:-1]):
            if b.offset:
                # yksk = b.joint_yksk(b.hmm.enumerate_support(expand=False).squeeze(-1), new_dim=True)
                yksk = b.joint_yksks0(b.hmm.enumerate_support(expand=False).squeeze(-1), new_dim=True).logsumexp(-2)
                yk = yksk.logsumexp(-1)
                # Compute the expectation of the cost function
                c = costs[-1] + yk
                c = c.logsumexp(-2)
                costs.append(c)
            else:
                # Gets the optimal permutation index.
                if is_cost_func:
                    c, perm_idx = costs[-1].min(-2)
                else:
                    c, perm_idx = costs[-1].max(-2)
                costs.append(c)
                perm_idxs.append(perm_idx)
        costs = costs[::-1]
        perm_idxs = perm_idxs[::-1]
        perm_tree = PermIdxTree(perm_idxs)
        if return_log_costs:
            return perm_tree, costs
        return perm_tree

    def prune_tree(self, idx):
        """Prunes a tree according to the index.

        :param idx: The index corresponding to the data or permutations.
        """
        idx = idx.unsqueeze(-1).unsqueeze(-2)
        new_tree = []
        for b in self.beliefs[1:]:
            idxb = torch.broadcast_tensors(idx, b.logits)[0]
            new_b = HMMBeliefState(b.logits.gather(0, idxb)[0], b.hmm, b.offset)
            new_tree.append(new_b)
        self.beliefs = new_tree


class PermIdxTree(object):

    def __init__(self, idx_list):
        self.perm_idxs = idx_list

    def trim_list_tree(self):
        r"""Trims the tree to remove permutation layers.

        The tree is a list of tensors. The first tensor is the root of the tree, and
        each subsequent tensor is a layer of the tree. The tree has a layered
        structure, with a path to a node in the tree given by the indices
        corresponding to the list :math:`(y_0, \sigma_0, y_1, \sigma_1, \ldots,)`,
        where :math:`y_i` is the index of the observation at step :math:`i`, and
        :math:`\sigma_i` is the index of the permutation at step :math:`i`.
        Once the permutations have been selected, the tree should be trimmed to
        remove the permutation layers, which is done by this function.
        """
        new_tree = []
        p = self.perm_idxs[0]
        p = p.squeeze()
        new_tree.append(p)
        for p in self.perm_idxs[1:]:
            p = p.squeeze()
            for ntp in new_tree:
                idx = torch.meshgrid([torch.arange(s) for s in ntp.shape])
                p = p[idx + (ntp,)]
            new_tree.append(p)
        self.perm_idxs = new_tree

    def expand_batch(self, data_len):
        r"""Adds a dimension of length data_len to each tensor in the tree.

        This function is used to expand the tree.

        :param int data_len: Length of new dimension.
        :return: Same list of tensors, but with a new dimension added to each
            tensor.
        """
        self.perm_idxs = [b.unsqueeze(-1).expand((-1,)*(len(b.shape)) + (data_len,)) for b in self.perm_idxs]

    def prune_perm_tree(self, data_idx):
        r"""Prunes the tree after observing data.

        Given data indexed by data_idx, this function prunes the tree to remove
        the branches that are not relevant to the data.

        :param torch.Tensor data_idx: Index of data.
        :return: Same list of tensors, but with the branches not relevant to the
            data removed.
        """
        # data_idx = data_idx.unsqueeze(-1)
        new_tree = []
        for pl in self.perm_idxs[1:]:
            new_b = pl[data_idx, ..., torch.arange(data_idx.shape[-1])]
            new_b = new_b.movedim(0, -1)
            new_tree.append(new_b)
        self.perm_idxs = new_tree
