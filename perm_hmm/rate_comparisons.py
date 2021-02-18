import torch
import pyro.distributions as dist

from perm_hmm.models.hmms import PermutedDiscreteHMM
from perm_hmm.simulator import HMMSimulator
from perm_hmm.classifiers.interrupted import IIDInterruptedClassifier, IIDBinaryIntClassifier
from perm_hmm.training.interrupted_training import exact_train_ic, train_ic, train_binary_ic, exact_train_binary_ic
from perm_hmm.util import num_to_data
from perm_hmm.postprocessing import ExactPostprocessor, EmpiricalPostprocessor
# from perm_hmm.loss_functions import log_zero_one, log_binary_zero_one


def exact_rates(phmm: PermutedDiscreteHMM, num_bins, perm_selector, classifier=None, num_ratios=20, verbosity=0, testing_states=None):
    # Initialize the parameters
    experiment_parameters = {
        b"hmm_params": {
            b"initial_logits": phmm.initial_logits,
            b"transition_logits": phmm.transition_logits,
            b"observation_params": phmm.observation_dist._param,
        },
        b"possible_perms": perm_selector.possible_perms,
        b"num_bins": torch.tensor(num_bins),
    }
    simulator = HMMSimulator(phmm)
    ic = IIDInterruptedClassifier(
        phmm.observation_dist,
        torch.tensor(1.),
        testing_states=testing_states,
    )
    base = len(phmm.observation_dist.enumerate_support())
    data = torch.stack(
        [num_to_data(num, num_bins, base) for num in range(base**num_bins)]
    ).float()
    lp = phmm.log_prob(data)
    plisd = phmm.posterior_log_initial_state_dist(data)
    log_joint = plisd.T + lp
    _ = exact_train_ic(ic, data, log_joint, num_ratios=num_ratios)
    nop = simulator.all_classifications(num_bins, classifier=classifier, verbosity=verbosity)
    pp = simulator.all_classifications(num_bins, classifier=classifier, perm_selector=perm_selector, verbosity=verbosity)
    if verbosity:
        nop, nod = nop
        pp, pd = pp
    i_results = ic.classify(data, verbosity=verbosity)
    if verbosity:
        i_classifications = i_results[0]
    else:
        i_classifications = i_results
    ip = ExactPostprocessor(log_joint, i_classifications)
    i_classifications = ip.classifications
    no_classifications = nop.classifications
    p_classifications = pp.classifications
    toret =  {
        # b"interrupted_log_rate": ip.log_risk(log_zero_one),
        b"interrupted_log_rate": ip.log_misclassification_rate(),
        b"permuted_log_rate": pp.log_misclassification_rate(),
        b"unpermuted_log_rate": nop.log_misclassification_rate(),
        b"interrupted_log_matrix": ip.log_confusion_matrix(),
        b"permuted_log_matrix": pp.log_confusion_matrix(),
        b"unpermuted_log_matrix": nop.log_confusion_matrix(),
        b"interrupted_classifications": i_classifications,
        b"unpermuted_classifications": no_classifications,
        b"permuted_classifications": p_classifications,
        b"experiment_parameters": experiment_parameters
    }
    if verbosity:
        toret[b"unpermuted_extras"] = nod
        toret[b"permuted_extras"] = pd
        toret[b"interrupted_break_ratio"] = i_results[1:]
    return toret


def empirical_rates(phmm: PermutedDiscreteHMM, num_bins, perm_selector, classifier=None, num_ratios=20, num_train=1000, num_samples=1000, confidence=.95, verbosity=0):
    experiment_parameters = {
        b"hmm_params": {
            b"initial_logits": phmm.initial_logits,
            b"transition_logits": phmm.transition_logits,
            b"observation_params": phmm.observation_dist._param,
        },
        b"possible_perms": perm_selector.possible_perms,
        b"num_bins": torch.tensor(num_bins),
    }
    simulator = HMMSimulator(phmm)
    ic = IIDInterruptedClassifier(
        phmm.observation_dist,
        torch.tensor(1.),
    )
    x, training_data = phmm.sample((num_train, num_bins))
    _ = train_ic(ic, training_data, x[..., 0], num_ratios=num_ratios)
    pp = simulator.simulate(num_bins, num_samples, classifier=classifier, perm_selector=perm_selector, verbosity=verbosity)
    nop, d = simulator.simulate(num_bins, num_samples, classifier=classifier, verbosity=max(1, verbosity))
    if verbosity:
        pp, pd = pp
    i_results = ic.classify(d[b"data"], verbosity=verbosity)
    if verbosity:
        i_classifications, i_dict = i_results
    else:
        i_classifications = i_results

    ip = EmpiricalPostprocessor(nop.ground_truth, i_classifications)

    i_classifications = ip.classifications
    no_classifications = nop.classifications
    p_classifications = pp.classifications
    toret = {
        b"interrupted_rates": ip.misclassification_rate(confidence),
        b"permuted_rates": pp.misclassification_rate(confidence),
        b"unpermuted_rates": nop.misclassification_rate(confidence),
        b"interrupted_classifications": i_classifications,
        b"unpermuted_classifications": no_classifications,
        b"permuted_classifications": p_classifications,
        b"experiment_parameters": experiment_parameters
    }
    if verbosity:
        toret[b"unpermuted_extras"] = d
        toret[b"permuted_extras"] = pd
        toret[b"interrupted_extras"] = i_dict
        toret[b"training_states"] = x
        toret[b"training_data"] = training_data
    return toret


def exact_binary_rates(phmm: PermutedDiscreteHMM, num_bins, perm_selector, dark_state, bright_state, classifier=None, num_ratios=20, verbosity=0):
    experiment_parameters = {
        b"hmm_params": {
            b"initial_logits": phmm.initial_logits,
            b"transition_logits": phmm.transition_logits,
            b"observation_params": phmm.observation_dist._param,
        },
        b"possible_perms": perm_selector.possible_perms,
        b"num_bins": torch.tensor(num_bins),
    }
    simulator = HMMSimulator(phmm)
    ic = IIDBinaryIntClassifier(
        type(phmm.observation_dist)(phmm.observation_dist._param[bright_state]),
        type(phmm.observation_dist)(phmm.observation_dist._param[dark_state]),
        torch.tensor(1.),
        torch.tensor(1.),
    )
    base = len(phmm.observation_dist.enumerate_support())
    data = torch.stack(
        [num_to_data(num, num_bins, base) for num in range(base**num_bins)]
    ).float()
    lp = phmm.log_prob(data)
    plisd = phmm.posterior_log_initial_state_dist(data)
    log_joint = plisd.T + lp
    _ = exact_train_binary_ic(ic, data, log_joint, dark_state, bright_state, num_ratios=num_ratios)
    nop = simulator.all_classifications(num_bins, classifier=classifier, verbosity=verbosity)
    pp = simulator.all_classifications(num_bins, classifier=classifier, perm_selector=perm_selector, verbosity=verbosity)
    if verbosity:
        nop, nod = nop
        pp, pd = pp
    i_results = ic.classify(data, verbosity=verbosity)
    if verbosity:
        i_classifications = i_results[0]
    else:
        i_classifications = i_results
    ip = ExactPostprocessor(log_joint, i_classifications)
    i_classifications = ip.classifications
    no_classifications = nop.classifications
    p_classifications = pp.classifications
    toret = {
        # b"interrupted_log_rate": ip.log_risk(log_binary_zero_one(dark_state, bright_state)),
        b"interrupted_log_rate": ip.log_misclassification_rate(),
        b"permuted_log_rate": pp.log_misclassification_rate(),
        b"unpermuted_log_rate": nop.log_misclassification_rate(),
        b"interrupted_log_matrix": ip.log_confusion_matrix(),
        b"permuted_log_matrix": pp.log_confusion_matrix(),
        b"unpermuted_log_matrix": nop.log_confusion_matrix(),
        b"interrupted_classifications": i_classifications,
        b"unpermuted_classifications": no_classifications,
        b"permuted_classifications": p_classifications,
        b"experiment_parameters": experiment_parameters
    }
    if verbosity:
        toret[b"unpermuted_extras"] = nod
        toret[b"permuted_extras"] = pd
        toret[b"interrupted_break_ratio"] = i_results[1:]
    return toret


def empirical_binary_rates(phmm: PermutedDiscreteHMM, num_bins, perm_selector, dark_state, bright_state, classifier=None, num_ratios=20, num_train=1000, num_samples=1000, confidence=.95, verbosity=0):
    experiment_parameters = {
        b"hmm_params": {
            b"initial_logits": phmm.initial_logits,
            b"transition_logits": phmm.transition_logits,
            b"observation_params": phmm.observation_dist._param,
        },
        b"possible_perms": perm_selector.possible_perms,
        b"num_bins": torch.tensor(num_bins),
    }
    simulator = HMMSimulator(phmm)
    ic = IIDBinaryIntClassifier(
        dist.Bernoulli(phmm.observation_dist._param[bright_state]),
        dist.Bernoulli(phmm.observation_dist._param[dark_state]),
        torch.tensor(1.),
        torch.tensor(1.),
        bright_state=bright_state,
        dark_state=dark_state,
    )
    x, training_data = phmm.sample((num_train, num_bins))
    _ = train_binary_ic(ic, training_data, x[..., 0], dark_state, bright_state, num_ratios=num_ratios)
    pp = simulator.simulate(num_bins, num_samples, classifier=classifier, perm_selector=perm_selector, verbosity=verbosity)
    nop, d = simulator.simulate(num_bins, num_samples, classifier=classifier, verbosity=max(1, verbosity))
    if verbosity:
        pp, pd = pp
    i_results = ic.classify(d[b"data"], verbosity=verbosity)
    if verbosity:
        i_classifications, i_dict = i_results
    else:
        i_classifications = i_results

    # HACK: We need to translate the binary classifications to the HMM classifications.
    # For purposes of computing misclassification rates, we can instead use the
    # binary_zero_one loss function, but I haven't implemented a confidence interval
    # for that, so we just do this instead.
    # testing_states = torch.tensor([dark_state, bright_state])
    # ip = EmpiricalPostprocessor(nop.ground_truth, testing_states[i_classifications.long()])
    ip = EmpiricalPostprocessor(nop.ground_truth, i_classifications)

    i_classifications = ip.classifications
    no_classifications = nop.classifications
    p_classifications = pp.classifications
    toret = {
        b"interrupted_rates": ip.misclassification_rate(confidence),
        b"permuted_rates": pp.misclassification_rate(confidence),
        b"unpermuted_rates": nop.misclassification_rate(confidence),
        b"interrupted_classifications": i_classifications,
        b"unpermuted_classifications": no_classifications,
        b"permuted_classifications": p_classifications,
        b"experiment_parameters": experiment_parameters
    }
    if verbosity:
        toret[b"unpermuted_extras"] = d
        toret[b"permuted_extras"] = pd
        toret[b"interrupted_extras"] = i_dict
        toret[b"training_states"] = x
        toret[b"training_data"] = training_data
    return toret

