import torch

from perm_hmm.models.hmms import PermutedDiscreteHMM
from perm_hmm.simulator import HMMSimulator
from perm_hmm.loss_functions import log_zero_one


def exact_rates(phmm: PermutedDiscreteHMM, num_steps, perm_policy, classifier=None, verbosity=0, log_loss=None):
    r"""Provides plumbing for comparing the misclassification rate calculated
    using a given policy versus using the trivial policy that applies no
    permutations.

    Basically calls :py:meth:`~perm_hmm.simulator.HMMSimulator.all_classifications`
    twice and packages the results into a dictionary.
    """
    experiment_parameters = {
        b"hmm_params": {
            b"initial_logits": phmm.initial_logits,
            b"transition_logits": phmm.transition_logits,
            b"observation_params": phmm.observation_dist._param,
        },
        b"possible_perms": perm_policy.possible_perms,
        b"num_steps": torch.tensor(num_steps),
    }
    simulator = HMMSimulator(phmm)
    nop = simulator.all_classifications(num_steps, classifier=classifier, verbosity=verbosity)
    pp = simulator.all_classifications(num_steps, classifier=classifier, perm_policy=perm_policy, verbosity=verbosity)
    if verbosity:
        nop, nod = nop
        pp, pd = pp
    no_classifications = nop.classifications
    p_classifications = pp.classifications
    if log_loss is None:
        log_loss = log_zero_one
    toret = {
        b"permuted_log_rate": pp.log_risk(log_loss),
        b"unpermuted_log_rate": nop.log_risk(log_loss),
        b"unpermuted_classifications": no_classifications,
        b"permuted_classifications": p_classifications,
        b"experiment_parameters": experiment_parameters
    }
    if verbosity:
        toret[b"unpermuted_extras"] = nod
        toret[b"permuted_extras"] = pd
    return toret


def empirical_rates(phmm: PermutedDiscreteHMM, num_steps, perm_policy, classifier=None, num_samples=1000, confidence=.95, verbosity=0, loss=None):
    experiment_parameters = {
        b"hmm_params": {
            b"initial_logits": phmm.initial_logits,
            b"transition_logits": phmm.transition_logits,
            b"observation_params": phmm.observation_dist._param,
        },
        b"possible_perms": perm_policy.possible_perms,
        b"num_steps": torch.tensor(num_steps),
    }
    simulator = HMMSimulator(phmm)
    pp = simulator.simulate(num_steps, num_samples, classifier=classifier, perm_policy=perm_policy, verbosity=verbosity)
    nop, d = simulator.simulate(num_steps, num_samples, classifier=classifier, verbosity=max(1, verbosity))
    if verbosity:
        pp, pd = pp

    no_classifications = nop.classifications
    p_classifications = pp.classifications
    toret = {
        b"permuted_rates": pp.misclassification_rate(confidence, loss),
        b"unpermuted_rates": nop.misclassification_rate(confidence, loss),
        b"unpermuted_classifications": no_classifications,
        b"permuted_classifications": p_classifications,
        b"experiment_parameters": experiment_parameters
    }
    if verbosity:
        toret[b"unpermuted_extras"] = d
        toret[b"permuted_extras"] = pd
    return toret
