import torch

from perm_hmm.classifiers.generic_classifiers import MAPClassifier


class PermClassifier(MAPClassifier):

    def __init__(self, model):
        self.model = model

    def classify(self, data, perms=None, verbosity=0):
        if perms is None:
            retval = super().classify(data, verbosity=verbosity)
        else:
            retval = MAPClassifier(self.model.expand_with_perm(perms)).classify(data, verbosity)
        return retval
