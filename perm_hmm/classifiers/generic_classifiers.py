class Classifier(object):

    def classify(self, data, verbosity=0):
        raise NotImplementedError

class MAPClassifier(Classifier):

    def __init__(self, model, testing_states):
        self.model = model
        self.testing_states = testing_states

    def classify(self, data, verbosity=0):
        plisd = self.model.posterior_log_initial_state_dist(data)
        classifications = self.testing_states[plisd[..., self.testing_states].argmax(-1)]
        if not verbosity:
            return classifications
        else:
            return {"classifications": classifications, "posterior_log_initial_state_dist": plisd}