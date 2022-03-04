import pytest
import torch
import pyro.distributions as dist

from perm_hmm.binning import bin_histogram, bin_log_histogram, binned_expanded_hmm, binned_hmm, optimally_binned_consecutive
from perm_hmm.models.hmms import DiscreteHMM, PermutedDiscreteHMM, ExpandedHMM
from example_systems.bin_beryllium import binned_hmm_class_value
from example_systems.beryllium import dimensionful_gamma, expanded_transitions, expanded_initial, expanded_outcomes


@pytest.mark.parametrize("base_hist,bin_edges,result", [
    (
            torch.tensor([
                [.1, .2, .3, .4],
                [.2, .5, 0., .3],
            ]),
            torch.tensor([
                    0, 2, 4
            ]),
            torch.tensor([
                    [.3, .7],
                    [.7, .3],
            ]),
    ),
    (
            torch.tensor([
                [.1, .9],
                [.2, .8],
            ]),
            torch.tensor([
                0, 1, 2
            ]),
            torch.tensor([
                [.1, .9],
                [.2, .8],
            ]),
    ),
])
def test_bin_histogram(base_hist, bin_edges, result):
    binned = bin_histogram(base_hist, bin_edges)
    assert binned.allclose(result)


@pytest.mark.parametrize("base_hist,bin_edges,result", [
    (
            torch.tensor([
                [.1, .2, .3, .4],
                [.2, .5, 0., .3],
            ]),
            torch.tensor([
                0, 2, 4
            ]),
            torch.tensor([
                [.3, .7],
                [.7, .3],
            ]),
    ),
    (
            torch.tensor([
                [.1, .9],
                [.2, .8],
            ]),
            torch.tensor([
                0, 1, 2
            ]),
            torch.tensor([
                [.1, .9],
                [.2, .8],
            ]),
    ),
])
def test_bin_log_histogram(base_hist, bin_edges, result):
    base_log_hist = torch.log(base_hist)
    log_result = torch.log(result)
    binned = bin_log_histogram(base_log_hist, bin_edges)
    assert binned.allclose(log_result)



def test_binned_hmm():
    n = 3
    il = (torch.ones(n)/n).log()
    transition_logits = (torch.eye(n) + 1e-14).log()
    transition_logits -= transition_logits.logsumexp(-1, keepdim=True)
    m = 8
    output_dirichlet = dist.Dirichlet(torch.ones(m)/m)
    observation_dist = dist.Categorical(probs=output_dirichlet.sample((n,)))
    hmm = DiscreteHMM(
        il,
        transition_logits,
        observation_dist,
    )
    bin_edges = torch.tensor([
        0, 3, 6, m
    ])
    binned = binned_hmm(hmm, bin_edges)
    assert (binned.transition_logits == hmm.transition_logits).all()
    assert (binned.initial_logits == hmm.initial_logits).all()


def test_optimally_binned_consecutive():
    n = 3
    il = (torch.ones(n)/n).log()
    transition_logits = (torch.eye(n) + 1e-14).log()
    transition_logits -= transition_logits.logsumexp(-1, keepdim=True)
    m = 8
    output_dirichlet = dist.Dirichlet(torch.ones(m)/m)
    observation_dist = dist.Categorical(probs=output_dirichlet.sample((n,)))
    hmm = PermutedDiscreteHMM(
        il,
        transition_logits,
        observation_dist,
    )
    min_edges, min_cost = optimally_binned_consecutive(hmm, 3)
    assert min_cost < torch.log(torch.tensor(.5))
    assert len(min_edges) == 4


def test_bin_beryllium():
    time = 1e-5 * dimensionful_gamma
    max_photons = 10
    num_bins = 4
    steps = 2
    v1 = binned_hmm_class_value(time, max_photons, num_bins, steps)
    initial_logits = expanded_initial(max_photons)
    transition_logits = expanded_transitions(time, max_photons)
    observation_dist = dist.Categorical(logits=torch.from_numpy(expanded_outcomes(max_photons)))
    expanded_hmm = ExpandedHMM(
        torch.from_numpy(initial_logits),
        torch.from_numpy(transition_logits),
        observation_dist,
    )
    v2 = optimally_binned_consecutive(expanded_hmm, num_bins, steps=steps)

    hmm1 = v1[b"hmm"]
    hmm2 = binned_expanded_hmm(expanded_hmm, v2[0])
    assert torch.allclose(hmm1.initial_logits.exp(), hmm2.initial_logits.exp())
    assert torch.allclose(hmm1.transition_logits.exp(), hmm2.transition_logits.exp())
    assert torch.allclose(hmm1.observation_dist._param.exp(), hmm2.observation_dist._param.exp())
