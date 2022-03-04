import numpy as np
from scipy.special import log1p, logsumexp
import matplotlib.pyplot as plt
from adapt_hypo_test.two_states import no_transitions as nt


def main():
    chis = []
    p = .09
    n = 10
    qs = np.arange(.1, .6, .01)
    for q in qs:
        sigmas, chi = nt.solve(p, q, n)
        chis.append(chi.ravel())
    chis = np.stack(chis)
    plt.plot(qs, chis[:, 0])
    plt.plot(qs, chis[:, 1])
    plt.show()
    r = log1p(-np.exp(logsumexp(chis, axis=-1) - np.log(2)))
    plt.plot(qs, r)
    plt.show()
    lps = np.arange(-8, 0, .1)
    lqs = np.arange(-4, 0, .1)
    xx, yy = np.meshgrid(lps, lqs)
    z = np.empty_like(yy)
    for i, lp in enumerate(lps):
        for j, lq in enumerate(lqs):
            _, chi = nt.solve(lp, lq, n, log=True)
            z[j, i] = log1p(-np.exp(logsumexp(chi.ravel(), axis=-1) - np.log(2)))
    z[xx > yy] = -float('inf')
    z[np.logaddexp(xx, yy) > 0] = -float('inf')
    plt.contourf(xx, yy, z)
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
