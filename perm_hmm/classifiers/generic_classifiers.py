class Classifier(object):
    """
    A generic classifier, has only the classify method.
    """

    def classify(self, data, verbosity=0):
        """Performs classification

        :param torch.Tensor data: Data to classify. Arbitrary shape.
        :param verbosity: Flag to return ancillary data generated in the computation.
        :return: If verbosity = 0, return just the classifications.
            Otherwise, return a tuple of length two. The first entry is the
            classifications, while the second is a dict.
        :raises NotImplementedError: If this method is not implemented.
        """
        raise NotImplementedError


class MAPClassifier(Classifier):
    """The `maximum a posteriori`_ classifier. Requires a model that implements
    posterior_log_initial_state_dist

    .. _`maximum a posteriori`: https://en.wikipedia.org/wiki/Maximum_a_posteriori_estimation

    Instances of this class have the following attributes:

    ``model``:
        A model that implements the method ``posterior_log_initial_state_dist``.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def classify(self, data, verbosity=0):
        """Classifies according to the maximum a posteriori classification.

        :param torch.Tensor data: Last dimension should be time.
        :param verbosity: Flag for whether to return the
            posterior log initial state distributions, used in the computation.
        :return: If verbosity = 0, the classifications, with shape data.shape[:-1]
            else, the classifications and a dictionary containing the posterior
            log initial state distribution, with key
            b"posterior_log_initial_state_dist".
        """
        plisd = self.model.posterior_log_initial_state_dist(data)
        classifications = plisd.argmax(-1)
        if not verbosity:
            return classifications
        else:
            return classifications, {b"posterior_log_initial_state_dist": plisd}
