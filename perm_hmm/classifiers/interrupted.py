"""
This module defines the interrupted classification scheme.

Using an iid model, we can make an inference based on data
which "collects enough evidence".
"""

import torch
from perm_hmm.util import first_nonzero, indices
from perm_hmm.classifiers.generic_classifiers import Classifier


class IIDInterruptedClassifier(Classifier):
    r"""
    A classifier which will terminate before the end of a run if a likelihood ratio exceeds a threshold.

    The classifier is initialized with a distribution which is a model of the distribution which generated the data :math:`p_i`,
    the states for which we are performing a multivalued hypothesis test for :math:`\mathcal{S}`,
    and possibly a log ratio :math:`R` which dictates whether or not the classifier terminates the run and concludes.

    At each time :math:`t`, denote by :math:`q^{(t)}_j(y^t)` the sorted likelihoods :math:`r_i(y^t) = \prod_{s=0}^tr_i(y_s)`,
    so that :math:`q^{(t)}_0(y^t) > \ldots > q^{(t)}_n(y^t)`. Then in particular :math:`q^{(t)}_0(y^t)` is the maximum likelihood of the data
    under the model. Then we compute
    .. math::

        \lambda_t = \log\Big(\frac{q^{(t)}_0(y^t)}{q^{(t)}_1(y^t)}\Big)

    If at any point we have :math:`lambda_t > R`, we terminate the run and make the inference
    .. math::

        \hat{s} = \mathcal{S}(\argmax{i} r_i(y^t))
    """

    def __init__(self, dist, ratio, testing_states=None):
        self.dist = dist
        """
        Distribution used to compute probabilities.
        Should have .batch_shape[-1] == num_states
        """
        self.ratio = ratio
        """
        Threshold likelihood ratio.
        """
        if testing_states is not None:
            self.testing_states = testing_states
        else:
            self.testing_states = torch.arange(self.dist.batch_shape[-1])

    def classify(self, data, verbosity=0):
        r"""
        Classifies data.

        At each time :math:`t`, denote by :math:`q^{(t)}_j(y^t)` the sorted likelihoods :math:`r_i(y^t) = \prod_{s=0}^tr_i(y_s)`,
        so that :math:`q^{(t)}_0(y^t) > \ldots > q^{(t)}_n(y^t)`. Then in particular :math:`q^{(t)}_0(y^t)` is the maximum likelihood of the data
        under the model. Then we compute
        .. math::

            \lambda_t = \log\Big(\frac{q^{(t)}_0(y^t)}{q^{(t)}_1(y^t)}\Big)

        If at any point we have :math:`lambda_t > R`, we terminate the run and make the inference
        .. math::

            \hat{s} = \mathcal{S}(\argmax{i} r_i(y^t))
        :param torch.Tensor data: Last dimension interpreted as time dimension.
        :param verbosity: If true, then return final log likelihood ratios and
            the break_flag, indicating whether or not the inference concluded
            before reaching the end of the time series.
        :return: If verbosity == 0, just the classifications, otherwise
            a tuple with the second entry a dict, containing

                b"break_flag": (Boolean tensor with shape == classifications.shape)
                    indicates if classification was performed before the end
                    of the time series

                b"log_like_ratio": (float tensor with shape == classifications.shape)
                    Final log likelihood ratio of the most likely to the second
                    most likely.

                b"sort_inds": (int tensor with shape == data.shape + (state_dim,))
                    Indices returned by torch.sort(intermediate_lps, -1),
                    Indicates order of likelihoods of states at that timestep.
        """
        shape = data.shape
        if shape == ():
            data = data.expand(1, 1)
        elif len(shape) == 1:
            data = data.expand(1, -1)
        data = data.float()
        intermediate_lps = self.dist.log_prob(data.unsqueeze(-1)).cumsum(dim=-2).float()
        sort_lps, sort_inds = torch.sort(intermediate_lps, -1)
        sort_lrs = sort_lps[..., -1] - sort_lps[..., -2]
        breaks = sort_lrs.view((1,)*len(self.ratio.shape) + sort_lrs.shape) > self.ratio.view(self.ratio.shape + (1,)*len(sort_lrs.shape))
        first_breaks = first_nonzero(breaks, -1)
        ix = indices(first_breaks.shape)
        _, sort_inds = torch.broadcast_tensors(breaks.unsqueeze(-1), sort_inds)
        classifications = sort_inds[..., -1, -1]
        mask = first_breaks < breaks.shape[-1]
        fb = first_breaks.clone().detach()
        fb[~mask] = -1
        classifications[mask] = sort_inds[ix + (fb, torch.zeros_like(fb, dtype=int))][mask]
        classifications = self.testing_states[classifications]
        if not verbosity:
            return classifications
        else:
            return classifications, {
                b"break_flag": breaks.any(-1),
                b"log_like_ratio": sort_lrs[..., -1],
                b"sort_inds": sort_inds,
                b"first_breaks": first_breaks,
            }


class IIDBinaryIntClassifier(Classifier):
    r"""
    Performs a classification between two states based on likelihood ratio tests,
    concluding if there is enough evidence.

    Distinct from :py:class:`IIDInterruptedClassifier` because there are two
    parameters to decide whether or not to interrupt the run.
    ..seealso:: :py:class:`IIDInterrupredClassifier`
    """

    def __init__(self, bright_model, dark_model, bright_ratio, dark_ratio, bright_state=None, dark_state=None):
        self.bright_model = bright_model
        self.dark_model = dark_model
        self.bright_ratio = bright_ratio
        r"""
        Torch float. Parameter such that if :math:`\log(L_{bright}/L_{dark})` exceeds
        it, the classifier concludes there is enough evidence to terminate and
        classify as bright
        """
        self.dark_ratio = dark_ratio
        r"""
        Torch float. Parameter such that if :math:`\log(L_{dark}/L_{bright})` exceeds
        it, the classifier concludes there is enough evidence to terminate and
        classify as dark
        """
        if (bright_state is not None) and (dark_state is not None):
            self.testing_states = torch.tensor([dark_state, bright_state])
        else:
            self.testing_states = None

    def classify(self, data, verbosity=0):
        r"""
        Performs classification.

        At each time :math:`t`, compute :math:`\lambda = log(L_{bright}(y^t)/L_{dark}(y^t))`,
        and conclude bright if :math:`\lambda` > self.bright_ratio, and conclude dark if
        :math:`-\lambda` > self.dark_ratio.
        :param data: Last dimension is interpreted as time.
        :param verbosity: Flag to indicate whether or not to return ancillary computations.
        :return: If verbosity == 0, returns classifications. Else, returns a tuple with
            first element the classifications and the second a dict containing

            b"break_flag": bool tensor, shape == classifications.shape
                Indicates if the classification concluded before reaching the
                end of the time series

            b"log_like_ratio" float tensor, shape == classifications.shape
                final :math:`log(L_{bright}(y^t)/L_{dark}(y^t))`.
        """
        shape = data.shape
        if shape == ():
            data = data.expand(1, 1)
        elif len(shape) == 1:
            data = data.expand(1, -1)
        data = data.float()

        intermediate_bright_lp = self.bright_model.log_prob(data).cumsum(dim=-1).float()
        intermediate_dark_lp = self.dark_model.log_prob(data).cumsum(dim=-1).float()
        intermediate_lr = intermediate_bright_lp - intermediate_dark_lp

        bright_most_likely = intermediate_lr[..., -1] > 0

        break_bright = intermediate_lr > self.bright_ratio
        break_dark = -intermediate_lr > self.dark_ratio

        first_break_bright = first_nonzero(break_bright, -1)
        first_break_dark = first_nonzero(break_dark, -1)
        bright_first = first_break_bright < first_break_dark

        bright_break_flag = break_bright.any(dim=-1)
        dark_break_flag = break_dark.any(dim=-1)
        break_flag = bright_break_flag | dark_break_flag
        neither_break = ~break_flag
        both_break = (bright_break_flag & dark_break_flag)
        one_break = bright_break_flag.logical_xor(dark_break_flag)

        classified_bright = \
            (one_break & bright_break_flag) | \
            (both_break & (bright_first & bright_break_flag)) | \
            (neither_break & bright_most_likely)

        classified_bright = classified_bright.long()
        if self.testing_states is not None:
            classifications = self.testing_states[classified_bright]
        else:
            classifications = classified_bright

        if not verbosity:
            return classifications
        else:
            return classifications, {
                b"break_flag": break_flag,
                b"log_like_ratio": intermediate_lr[..., -1],
                b"first_break_bright": first_break_bright,
                b"first_break_dark": first_break_dark,
            }
