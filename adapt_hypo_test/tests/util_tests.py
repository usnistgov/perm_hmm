import pytest
from adapt_hypo_test.two_states.util import *


@pytest.mark.parametrize("p",[
    (.1,),
    (np.arange(0, 1, .1))
])
def test_log_odds_to_log_probs(p):
    p = .1
    x = np.log(p/(1-p))
    lps = log_odds_to_log_probs(x)
    lps2 = np.log([1-p, p])
    assert np.allclose(lps, lps2)


#%%
@pytest.mark.parametrize('p, q',
                         np.random.random((100, 2)))
def test_log_p_log_q_to_m(p, q):
    m1 = pq_to_m(p, q)
    m2 = log_p_log_q_to_m(np.log(p), np.log(q))
    assert np.allclose(m1, m2)


