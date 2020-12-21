"""
This module defines the interrupted classification scheme, which contrasts with
the schemes in the :py:mod:`perm_hmm.perm_hmm.bayesian_tracking`
module.
Using a simple model with no transitions, we can make an inference based on data
which "collects enough evidence".
"""

import torch
from perm_hmm.util import first_nonzero, indices
from perm_hmm.return_types import ClassBreakRatio
from perm_hmm.classifiers.generic_classifiers import Classifier


class InterruptedClassifier(Classifier):
    r"""
    A classifier which will terminate before the end of a run if a likelihood ratio exceeds a threshold.

    The classifier is initialized with a distribution which is a model of the distribution which generated the data :math:`p_i`,
    the states for which we are performing a multivalued hypothesis test for :math:`\mathcal{S}`,
    and possibly a log ratio :math:`R` which dictates whether or not the classifier terminates the run and concludes.

    Relabel the model so that we only consider the states which are being tested. :math:`r_i = p_{\mathcal{S}(i)}`.
    At each time :math:`t`, denote by :math:`q^{(t)}_j(y^t)` the sorted likelihoods :math:`r_i(y^t) = \prod_{s=0}^tr_i(y_s)`,
    so that :math:`q^{(t)}_0(y^t) > \ldots > q^{(t)}_n(y^t)`. Then in particular :math:`q^{(t)}_0(y^t)` is the maximum likelihood of the data
    under the model. Then we compute
    .. math::

        \lambda_t = \log\Big(\frac{q^{(t)}_0(y^t)}{q^{(t)}_1(y^t)}\Big)

    If at any point we have :math:`lambda_t > R`, we terminate the run and make the inference
    .. math::

        \hat{s} = \mathcal{S}(\argmax{i} r_i(y^t))
    """

    def __init__(self, dist, testing_states, ratio=None):
        # TODO: Fix this class and its trainer.
        self.dist = dist
        self.testing_states = testing_states
        self.ratio = ratio

    def classify(self, data, ratio=None):
        if ratio is None:
            if self.ratio is None:
                raise ValueError("Must train before classifying.")
            ratio = self.ratio
        shape = data.shape
        if shape == ():
            data = data.expand(1, 1)
        elif len(shape) == 1:
            data = data.expand(1, -1)
        data = data.float()
        intermediate_lps = self.dist.log_prob(data.unsqueeze(-1))[..., self.testing_states].cumsum(dim=-2).float()
        sort_lps, sort_inds = torch.sort(intermediate_lps, -1)
        sort_lrs = sort_lps[..., -1] - sort_lps[..., -2]
        breaks = sort_lrs.view((1,)*len(ratio.shape) + sort_lrs.shape) > ratio.view(ratio.shape + (1,)*len(sort_lrs.shape))
        first_breaks = first_nonzero(breaks, -1)
        ix = indices(first_breaks.shape)
        _, sort_inds = torch.broadcast_tensors(breaks.unsqueeze(-1), sort_inds)
        classifications = sort_inds[..., -1, -1]
        classifications[first_breaks >= 0] = sort_inds[ix + (first_breaks, torch.zeros_like(first_breaks, dtype=int))][first_breaks >= 0]
        classifications = self.testing_states[classifications]
        return ClassBreakRatio(
            classifications,
            breaks.any(-1),
            sort_lrs[..., -1],
        )
