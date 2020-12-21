"""
Provides the functions for computing exact and approximate misclassification
rates of various repeated measurement schemes. Aims to demonstrate when a
strategy which involves applying permutations between observations yields an
appreciable advantage.
"""

import perm_hmm.strategies
import perm_hmm.strategies.min_ent
import perm_hmm.util
import perm_hmm.hmms
