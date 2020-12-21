import torch

from perm_hmm.classifiers.generic_classifiers import MAPClassifier


class PermClassifier(MAPClassifier):

    def __init__(self, model, testing_states):
        self.model = model
        self.testing_states = testing_states

    def classify(self, data, perms=None, verbosity=0):
        retval = MAPClassifier(self.model.expand_with_perm(perms), self.testing_states).classify(data, verbosity)
        return retval
