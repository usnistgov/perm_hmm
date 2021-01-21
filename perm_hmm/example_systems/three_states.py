import numpy as np

num_states = 3


def gen_hmm(a, b, c=None, d=None):
    if c is None:
        c = b
    if d is None:
        d = a
    initial_probs = np.ones(num_states)/num_states
    output_probs = np.array([[1-a, a, 0.], [(1-d)/2, d, (1-d)/2], [0., a, (1-a)-0.]])
    transition_probs = np.array([[1-b-0., b, 0.],[(1-c)/2, c, (1-c)/2],[0., b, 1-b-0.]])
    return initial_probs, transition_probs, output_probs

