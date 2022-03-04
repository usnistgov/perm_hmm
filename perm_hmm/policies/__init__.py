r"""
This module contains classes that select permutations for the HMM.

The base class is
:py:class:`~perm_hmm.policies.policy.PermPolicy`,
which should be subclassed to make a custom
policy. A simple example of a policy is given in
:py:class:`~perm_hmm.policies.rotator_policy.RotatorPolicy`.
The :py:class:`~perm_hmm.policies.belief.BeliefStatePolicy` is a more
complex subclass of policy that uses the
:py:class:`~perm_hmm.belief.HMMBeliefState` to select the next permutation.

A different subclass that uses the :py:class:`~perm_hmm.belief.HMMBeliefState`
is the :py:class:`~perm_hmm.policies.min_tree.MinTreePolicy`, which
computes the next permutation by looking ahead some number of steps, and
choosing the one that minimizes the expected cost.

The :py:class:`~perm_hmm.policies.exhaustive.ExhaustivePolicy` is a policy
that chooses the optimal permutation by exhaustive search. This is generically
extremely expensive, costing O((n*p)**t) time, where n is the number of
outcomes, p is the number of permutations, and t is the number of steps.

The :py:class:`~perm_hmm.policies.min_ent.MinEntropyPolicy` is a policy
that chooses the next permutation by minimizing the expected posterior entropy
of the initial state, based on one step look-ahead. It is a subclass of the
:py:class:`~perm_hmm.policies.min_tree.MinTreePolicy`.

Finally, the submodule :py:mod:`~perm_hmm.tree_strategy_funcs` contains utility
functions used by the policies based on tree-based strategies.
"""

from .policy import PermPolicy
from .belief import BeliefStatePolicy
