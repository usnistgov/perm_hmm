from perm_hmm.classifiers.generic_classifiers import MAPClassifier


class PermClassifier(MAPClassifier):
    """
    MAP classifier for an HMM with permutations.
    """

    def classify(self, data, perms=None, verbosity=0):
        """
        Classifies data.
        :param torch.Tensor data: To be classified. Last dimension interpreted as time.
        :param perms: Permutations. Should have shape == data.shape + (num_states,)
        :param verbosity: If nonzero, returns a tuple with second element a dict
            containing key b"posterior_log_initial_state_dist".
        :return: Classifications, and if verbosity is nonzero, a dict as well.
        """
        if perms is None:
            retval = super().classify(data, verbosity=verbosity)
        else:
            retval = MAPClassifier(self.model.expand_with_perm(perms)).classify(data, verbosity)
        return retval
