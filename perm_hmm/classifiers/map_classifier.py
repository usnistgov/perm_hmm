from perm_hmm.hmms import SampleableDiscreteHMM

class MAPClassifier(object):

    def __init__(self, hmm: SampleableDiscreteHMM, testing_states):
        self.hmm = hmm
        self.testing_states = testing_states

    def classify(self, data):
        plisd = self.hmm.posterior_log_initial_state_dist(data)
        classifications = self.testing_states[plisd[..., self.testing_states].argmax(-1)]
        return classifications
