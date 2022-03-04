"""
Provides the functions for computing exact and approximate misclassification
rates of various repeated measurement schemes. Aims to demonstrate when a
strategy which involves applying permutations between observations yields an
appreciable advantage.
"""

import perm_hmm.policies
import perm_hmm.policies.min_tree
import perm_hmm.util
import perm_hmm.models.hmms
