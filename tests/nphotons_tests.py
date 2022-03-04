import pytest
import numpy as np
from scipy.special import logsumexp
import example_systems.beryllium as beryllium


@pytest.mark.parametrize("time", [
    1e-7, 1e-6, 1e-5, 1e-4
])
def test_prob_of_n_photons(time):
    integration_time = beryllium.dimensionful_gamma * time
    pn0 = np.exp(beryllium.log_prob_n_given_l(0, integration_time))
    p0 = beryllium.log_prob_l_zero_given_lp(integration_time)
    p0 = logsumexp(p0, axis=-1)
    p0 = np.exp(p0)
    assert np.allclose(p0, pn0)
