import torch

from perm_hmm.classifiers.generic_classifiers import MAPClassifier


class PermClassifier(MAPClassifier):

    def __init__(self, model):
        self.model = model

    def classify(self, data, testing_states, perms=None, verbosity=0):
        retval = MAPClassifier(self.model.expand_with_perm(perms)).classify(data, testing_states, verbosity)
        return retval
