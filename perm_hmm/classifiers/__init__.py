"""
Classifiers built from models in perm_hmm.

:py:class:`~perm_hmm.classifiers.generic_classifiers.MAPClassifier`
is a maximum a posteriori classifier.

:py:class:`~perm_hmm.classifiers.perm_classifier.PermClassifier`
Uses permutations to compute the posterior log initial state distributions, then
computes the classifications using the Maximum a posteriori classifier.
"""
