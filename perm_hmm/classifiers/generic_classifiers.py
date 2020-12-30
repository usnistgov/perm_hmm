class Classifier(object):

    def classify(self, data, verbosity=0):
        raise NotImplementedError

class MAPClassifier(Classifier):

    def __init__(self, model):
        self.model = model

    def classify(self, data, verbosity=0):
        plisd = self.model.posterior_log_initial_state_dist(data)
        classifications = plisd.argmax(-1)
        if not verbosity:
            return classifications
        else:
            return {b"classifications": classifications, b"posterior_log_initial_state_dist": plisd}