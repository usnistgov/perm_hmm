r"""Computes belief states of HMMs with permutations.

This module contains the :py:class:`~perm_hmm.policies.belief.HMMBeliefState`
class, which computes belief states of HMMs with permutations in a tree-like
manner.

This module also contains the
:py:class:`~perm_hmm.policies.belief.BeliefStatePolicy` class, which is an
abstract class that shows how to generically use the HMMBeliefState for
computing permutations.
"""
from copy import deepcopy
import torch
from perm_hmm.util import ZERO
from perm_hmm.policies.policy import PermPolicy


class HMMBeliefState(object):
    r"""Stores the current belief state of the HMM, and provides update methods.

    The HMMBeliefState has an attribute ``.logits`` that is a tensor of shape
    (n_outcomes, n_permutations, n_outcomes, ...) + (n_batch, n_states,
    n_states).  The first part of the shape is the tree, while the second part
    is the belief state attached to each leaf. The path from the root to a leaf
    is indexed by the sequence :math:`(y_0, \sigma_0, y_1, \sigma_1, ...)`,
    where the :math:`y_i` are the outcomes at the :math:`i`-th step, and the
    :math:`\sigma_i` are the permutations at the :math:`i`-th step.  Because of
    the layered structure of this tree, the leaves can be indexed with either an
    outcome or a permutation. To account for this possibility, the BeliefPolicy
    class comes with a flag called ``.offset`` that is true if and only if the
    leaves of the tree are labelled by outcomes. The reason for the name offset
    is because if the leaves of the tree are outcomes, then the belief states
    are :math:`\mathbb{P}(s_0, s_{k+1}|y^k)`, while otherwise it is
    :math:`\mathbb{P}(s_0, s_k|y^k)`, so that the "current" state step index is
    offset from the "current" data step index.

    Instances of this class have the following attributes:

    ``logits``:
        :py:class:`~torch.Tensor` that stores the log probabilities
        :math:`\mathbb{P}(s_0, s_k|y^k)` if `offset` is `False`,
        or :math:`\mathbb{P}(s_0, s_k|y^{k-1})`  when it is `True`.
        Dimension -1 is the :math:`s_k` dimension, -2 is the
        :math:`s_0` dimension, and -3 is the :math:`y^k` dimension.

    ``offset``:
        A boolean indicating whether this belief state has the :math:`y` step
        index offset from the :math:`s` step index. This is necessary because
        the updating of the belief state takes place in two steps: We update the
        belief state based on new data, and we update according to a chosen
        permutation that induces a particular transition matrix. The Bayes' rule
        update is only possible when `offset` is True, and updates the belief
        state from :math:`\mathbb{P}(s_0, s_k|y^{k-1})` to
        :math:`\mathbb{P}(s_0, s_k|y^k)`, while the transition update is only
        possible when `offset` is False, and updates the belief state from
        :math:`\mathbb{P}(s_0, s_k|y^k)` to
        :math:`\mathbb{P}(s_0, s_{k+1}|y^k)`.

    ``hmm``:
        A :py:class:`~perm_hmm.models.hmms.PermutedDiscreteHMM` that is used
        to update the belief state.
    """

    def __init__(self, logits, hmm, offset=True):
        r"""Initializes the object.

        Stores the distribution
        :math:`\mathbb{P}(s_0, s_k|y^{k-1})` if `offset=True`,
        and :math:`\mathbb{P}(s_0, s_k|y^k)` if `offset=False`.

        Dimension -1 is the :math:`s_k` dimension, -2 is the
        :math:`s_0` dimension, and -3 is the :math:`y^k` dimension.

        We opt to put :math:`y^k` in the -3 dimension rather than the 0
        dimension because it often needs to be updated.

        :param torch.Tensor logits: A tensor of shape (..., n_states, n_states).
        :param DiscreteHMM hmm: The HMM to use for updating the belief state.
        :param bool offset: Whether the leaves of the tree are labelled by
            outcomes or permutations.
        """
        self.logits = logits  # type: torch.Tensor
        # Offset is true if state is :math:`\mathbb{P}(s_0, s_k|y^{k-1})`,
        # and false if it is :math:`\mathbb{P}(s_0, s_k|y^k)`.
        self.offset = offset
        self.hmm = hmm

    @classmethod
    def from_hmm(cls, hmm):
        r"""A factory method that constructs an HMMBeliefState from an HMM.

        Outputs the belief state
        :math:`\mathbb{P}(s_0, s_0'|y^{-1}) = \delta_{s_0, s_0'}`, where
        the data :math:`y^{-1}` is the empty tuple.

        :param perm_hmm.models.hmms.PermutedDiscreteHMM hmm: A DiscreteHMM.
        :return: An HMMBeliefState with offset = True
        """
        logits = hmm.initial_logits.clone().detach()
        val = torch.full((logits.shape[0], logits.shape[0]), ZERO, dtype=logits.dtype).log()
        val[torch.arange(logits.shape[0]), torch.arange(logits.shape[0])] = logits
        val -= val.logsumexp(-2, keepdim=True).logsumexp(-1, keepdim=True)
        offset = True
        return cls(val, hmm, offset)

    @classmethod
    def from_skipfirsthmm(cls, skipfirsthmm, trivial_obs=None):
        r"""A factory method that constructs an HMMBeliefState from a
            :py:class:`~perm_hmm.models.hmms.SkipFirstDiscreteHMM`.

        The distinction with
        :py:meth:`~perm_hmm.policies.belief.from_hmm`
        is that, because of the encoding of the SkipFirstDiscreteHMM as having
        the first outcome being a dummy outcome, we need to perform an extra
        transition before seeing any data.

        :param perm_hmm.models.hmms.SkipFirstDiscreteHMM skipfirsthmm:
        :param torch.Tensor trivial_obs: The observation to use for the first
            bayes update.
        :return: :math:`\mathbb{P}(s_0, s_1|y^{-1}) = A_{s_0, s_1}`, where
            A is the transition matrix of the SkipFirstDiscreteHMM.
        """
        b = HMMBeliefState.from_hmm(skipfirsthmm)
        if trivial_obs is None:
            trivial_obs = torch.zeros((1,) + skipfirsthmm.observation_dist.event_shape[:1])
        if not trivial_obs.shape:
            trivial_obs = trivial_obs.unsqueeze(-1)
        b = b.bayes_update(trivial_obs)
        trivial_perm = torch.arange(skipfirsthmm.initial_logits.shape[-1])
        b = b.transition(trivial_perm)
        return b

    @classmethod
    def from_expandedhmm(cls, expandedhmm, trivial_obs=None,
                         initial_states_to_keep=None):
        if initial_states_to_keep is None:
            initial_states_to_keep = ~torch.isclose(
                expandedhmm.initial_logits.exp(), torch.tensor(0, dtype=float), atol=1e-7)
        b = cls.from_skipfirsthmm(expandedhmm, trivial_obs=trivial_obs)
        b.logits = b.logits[..., initial_states_to_keep, :]
        return b

    def joint_yksks0(self, obs, observation_dist=None, new_dim=False):
        r"""Computes :math:`\mathbb{P}(s_0, s_k, y_k|y^{k-1})`

        This method is used to compute the joint probability of the
        initial state, the current state, and the current observation.

        :param torch.Tensor obs: The observed data.
        :param observation_dist: Defaults to self.hmm.observation_dist
        :param bool new_dim: Indicates whether to add a new dimension to
            the output. If specified, the output will add a new dimension of
            length equal to that of obs, in position -4.
        :return: A tensor containing the  joint distribution, with dimensions
            -1: s_k, -2: s_0, -3: y_k if `new_dim=False`, and batch otherwise,
            -4: y_k if `new_dim=True`.
        """
        if not self.offset:
            raise ValueError("Cannot compute joint distribution if offset is False.")
        if observation_dist is None:
            observation_dist = self.hmm.observation_dist
        lls = observation_dist.log_prob(obs.unsqueeze(-1))
        # Make space for initial state
        lls = lls.unsqueeze(-2)
        v = self.logits
        if new_dim:
            v = v.unsqueeze(-3)
        v = v + lls
        if new_dim:
            if len(v.shape) == 3:
                v = v.unsqueeze(-4)
            return v.transpose(-3, -4)
        return v

    def joint_yksk(self, obs, observation_dist=None, new_dim=False):
        r"""Computes :math:`\mathbb{P}(s_k, y_k|y^{k-1})`

        This method is used to compute the joint probability of the current
        state and the current observation.  This method is necessary because
        just using the
        :py:meth:`~perm_hmm.policies.belief.HMMBeliefState.joint_yksks0` method
        does unnecessary work if we only want
        :math:`\mathbb{P}(y_k, s_k|y^{k-1})`.

        :param obs: The observed data.
        :param observation_dist: The observation distribution to update the
            belief state with. Defaults to ``self.hmm.observation_dist``
        :param bool new_dim: Indicates whether to add a new dimension to
            the output. If specified, the output will add a new dimension of
            length equal to that of obs, in position -3.
        :return: The joint current state and observation distribution, with
            dimensions -1: s_k, -2: y_k if `new_dim=False`, and batch otherwise,
            -3: y_k if `new_dim=True`.
        """
        if not self.offset:
            raise ValueError("Cannot compute joint distribution if offset is False.")
        if observation_dist is None:
            observation_dist = self.hmm.observation_dist
        lls = observation_dist.log_prob(obs.unsqueeze(-1))
        # Marginalize over initial state
        v = self.logits.logsumexp(-2)
        if new_dim:
            v = v.unsqueeze(-2)
        v = v + lls
        if new_dim:
            if len(v.shape) == 2:
                v = v.unsqueeze(-3)
            return v.transpose(-2, -3)
        return v

    def bayes_update(self, obs, hmm=None, new_dim=False):
        r"""Starting from :math:`\mathbb{P}(s_0, s_k|y^{k-1})`, we update the
        belief state to :math:`\mathbb{P}(s_0, s_k|y^k)` using Bayes' rule.

        This method is variously used to expand the leaves of the tree, or to
        update the tree. This method is a constructor-style method, meaning
        it returns a new object. We make new objects because sometimes it is
        useful to have both the original and the updated belief state.

        :param obs: The observed data.
        :param hmm: The hmm containing the distribution to update with.
            Defaults to self.hmm
        :param bool new_dim: Indicates whether to add a new dimension to
            the output. If specified, the output will add a new dimension of
            length equal to that of obs, in position -4.
        :return: A new :py:class:`~perm_hmm.policies.belief.HMMBeliefState`
            object with .offset=False, and .logits = the updated belief state.
            .logits has dimensions -1: s_k, -2: s_0, -3: batch, and -4: new
            dimension corresponding to obs if new_dim=True.
        """
        if hmm is None:
            hmm = self.hmm
        v = self.joint_yksks0(obs, observation_dist=hmm.observation_dist, new_dim=new_dim)
        v -= v.logsumexp(-1, keepdim=True).logsumexp(-2, keepdim=True)
        return self.__class__(v, hmm, offset=False)

    def transition(self, perm, hmm=None, new_dim=False):
        r"""Starting from :math:`\mathbb{P}(s_0, s_k|y^k)`, we update the
        belief state to :math:`\mathbb{P}(s_0, s_{k+1}|y^k)` using
        the transition matrix of the input hmm.

        This method is variously used to expand the leaves of the tree, or to
        update the tree. This method is a constructor-style method, meaning
        it returns a new object. We make new objects because sometimes it is
        useful to have both the original and the updated belief state.

        :param perm: The observed data.
        :param hmm: The hmm containing the log transition matrix to use.
            Defaults to self.hmm
        :param bool new_dim: Indicates whether to add a new dimension to
            the output. If specified, the output will add a new dimension of
            length equal to that of perm, in position -4.
        :return: A new :py:class:`~perm_hmm.policies.belief.HMMBeliefState`
            with .offset=True, and .logits = the updated belief state. The
            logits have dimensions -1: s_k, -2: s_0, -3: batch, and -4: new
            dimension corresponding to perm if new_dim=True.
        """
        if self.offset:
            raise ValueError("Cannot transition belief state if offset is True.")
        # Make space for initial state
        if hmm is None:
            hmm = self.hmm
        transition_logits = hmm.transition_logits[perm].unsqueeze(-3)
        # Unsqueeze -1 for state :math:`s_{k+1}`,
        # unsqueeze -4 for perm choice.
        # Logsumexp -2 to marginalize :math:`s_k`.
        v = (self.logits.unsqueeze(-1).unsqueeze(-4) + transition_logits).logsumexp(-2)
        if new_dim:
            return self.__class__(v.transpose(-3, -4), hmm, offset=True)
        return self.__class__(v.squeeze(-3), hmm, offset=True)


class BeliefStatePolicy(PermPolicy):
    r"""Abstract PermPolicy class for selecting a permutation based on the
    most up-to-date belief state.

    Generically, one wants to make a policy based on the most updated belief
    state. This class takes care of all the various updating necessary. One can
    subclass this class and write a method calculate_perm that calculates the
    permutation based on the beliefs :math:`\mathbb{P}(s_0, s_k|y^{k})`.

    In addition to the attributes of the base class, instances of this class
    have the following attributes:

    ``hmm``:
        A :py:class:`~perm_hmm.models.hmms.PermutedDiscreteHMM` that is used to
        update the belief state.

    ``belief_state``:
        A :py:class:`~perm_hmm.policies.belief.HMMBeliefState` that represents
        the current belief.
    """

    def __init__(self, possible_perms, hmm, save_history=False):
        r"""Initializes the policy.

        .. seealso:: :py:meth:`~perm_hmm.policies.policy.PermPolicy.__init__`

        :param possible_perms: The possible permutations.
        :param hmm: The HMM used to compute the outcome probabilities.
        :param save_history: Whether to save the computation history the
            next time that the policy is called. This history is saved in the
            attribute .calc_history.
        """
        super().__init__(possible_perms, save_history=save_history)
        self.hmm = deepcopy(hmm)
        self.belief_state = HMMBeliefState.from_hmm(hmm)

    def reset(self, save_history=False):
        r"""Resets the policy, and in particular resets the belief state.

        .. seealso:: :py:meth:`~perm_hmm.policies.policy.PermPolicy.reset`

        :param bool save_history: Whether to save the computation history
            the next time that the policy is used.
        :return: None
        """
        super().reset(save_history=save_history)
        self.belief_state = HMMBeliefState.from_hmm(self.hmm)

    def calculate_perm_from_belief(self, return_dict=False):
        r"""The method that calculates the permutation based on the most
            up-to-date belief state.

        This method should be overwritten by subclasses.

        For an example implementation, see the method from the MinTreePolicy.
        .. seealso:: :py:meth:`~perm_hmm.policies.min_tree.MinTreePolicy.calculate_perm`

        :param bool return_dict: Whether to return a dictionary of the
            computation history.
        :return: The permutation, and if return_dict=True, a dictionary of the
            computation history.
        """
        raise NotImplementedError

    def bayes_update(self, obs):
        r"""Updates the belief state using Bayes' rule.

        Uses the HMM to update the belief state.

        .. seealso:: :py:meth:`~perm_hmm.policies.belief.HMMBeliefState.bayes_update`

        :param obs: The observation to update the belief state with.
        :return: None
        """
        self.belief_state = self.belief_state.bayes_update(obs)

    def transition(self, perm):
        r"""Updates the belief state using the transition probabilities, and the
            selected permutation.

        Uses the HMM and the selected permutation to update the belief state.

        .. seealso:: :py:meth:`~perm_hmm.policies.belief.HMMBeliefState.transition`

        :param perm: The permutation used.
        :return: None
        """
        self.belief_state = self.belief_state.transition(perm.unsqueeze(-2))

    def calculate_perm(self, obs, event_dims=0):
        r"""Generates a permutation based on the most up-to-date belief state.

        This method first updates the belief state using the observation, then
        calculates the permutation based on the new belief state, using the
        method
        :py:meth:`~perm_hmm.policies.belief.BeliefStatePolicy.calculate_perm_from_belief`.
        After the permutation is calculated, the belief state is updated using
        the permutation, via the method
        :py:meth:`~perm_hmm.policies.belief.BeliefStatePolicy.transition`.
        Then, the permutation is returned, along with the computation history.

        :param obs: The observation to update the belief state with.
        :param event_dims: The number of event dimensions. Used to distinguish
            batch dimensions from the event dimensions.
        :return: The permutation and the computation history.
        """
        self.bayes_update(obs)
        perm, diction = self.calculate_perm_from_belief(return_dict=True)
        self.transition(perm)
        diction[b'belief'] = self.belief_state.logits.clone().detach()
        return perm, diction

    def _calculate_beliefs(self, data, perms):
        r"""Given a set of data and a set of perms, calculates the belief states
        for the sequence.

        This method is useful to calculate the belief states for a sequence of
        data and permutations.

        :param data: The data to calculate the belief states for.
        :param perms: The permutations that follow the data. Must have the same
            batch shape as data.
        :raises: ValueError if the batch shapes of data and perms are not the
            same.
        :raises: ValueError if the data is not of a shape that is compatible
            with the HMM.
        :return: The belief states.
        """
        shape = perms.shape[:-1]
        if not data.shape[:len(shape)] == shape:
            raise ValueError("Data and permutations must have same batch shape, but got {} and {}".format(data.shape[:len(shape)], shape))
        try:
            _ = self.hmm.log_prob(data)
        except (ValueError, RuntimeError) as e:
            raise ValueError("Data does not have a compatible shape") from e

        sel = FixedPolicy(self.possible_perms, self.hmm, perms, save_history=True)

        sel.get_perms(data)
        return sel.calc_history[b"belief"]


class FixedPolicy(BeliefStatePolicy):
    r"""A policy that always returns the same sequence of permutations.

    Used to compute the belief states for a sequence of data and permutations.
    In addition to the attributes of the base class, this class has the
    attribute ``perms``, which is the permutations that will be returned, and
    the attribute ``step``, which is the number of permutations that have been
    returned for the current sequence. This is reset to 0 in the reset method.

    In addition to the attributes of the base class, instances of this class
    have the following attributes:

    ``step``:
        An :py:class:`~int` that indiccates how many steps have passed.

    ``perms``:
        A :py:class:`~torch.Tensor` that is the fixed set of permutations to be
        used.
    """

    def __init__(self, possible_perms, hmm, perms, save_history=False):
        r"""Initializes the policy.

        Needs the ``perms`` argument, that is the permutations that will be
        returned, independent of the data.

        :param torch.Tensor possible_perms: The allowed permutations.
        :param perm_hmm.models.hmms.PermutedDiscreteHMM hmm:
            The HMM used to calculate the belief states.
        :param torch.Tensor perms: The fixed sequence of permutations to be returned.
        :param bool save_history: Whether to save the computation history.
        """
        super().__init__(possible_perms, hmm, save_history=save_history)
        self.step = 0
        self.perms = perms

    def reset(self, save_history=False):
        r"""Resets the policy.

        Resets the step to 0, and the computation history.

        :param bool save_history: Whether to save the computation history.
        :return: None
        """
        super().reset(save_history=save_history)
        self.step = 0

    def calculate_perm_from_belief(self, return_dict=False):
        r"""Trivial implementation of the base class method.

        Returns the permutation corresponding to the current step, and an empty
        dictionary for the computation history if ``return_dict`` is True.

        :param bool return_dict: Whether to return the computation history.
        :return: The permutation and, if ``return_dict`` is True, the
            computation history.
        """
        p = self.perms[..., self.step, :]
        self.step += 1
        if return_dict:
            return p, {}
        return p
