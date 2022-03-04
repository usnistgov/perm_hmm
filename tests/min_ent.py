"""Conditioned on the data seen thus far, computes the expected posterior
entropy of the initial state, given the yet to be seen next data point, in
expectation. This computation is done for each allowed permutation. Then
minimizing the computed quantity over permutations, we obtain the permutation
to apply.

"""
from perm_hmm.models.hmms import SkipFirstDiscreteHMM
from perm_hmm.policies.belief import BeliefStatePolicy, HMMBeliefState


class MinEntropyPolicy(BeliefStatePolicy):

    def distributions_for_all_perms(self):
        r"""
        Returns :math:`\mathbb{P}(s_0, s_k, y_k|y^{k-1})` for all possible
        permutations.

        Dimensions: -1: y_k, -2: s_k, -3: s_0, -4: permutations, -5: y^{k-1}
        :return:
        """
        state = self.belief_state.transition(self.possible_perms)
        all_obs = self.hmm.observation_dist.enumerate_support(expand=False)
        lls = self.hmm.observation_dist.log_prob(all_obs)
        lls = lls.T
        distn = state.logits.unsqueeze(-1) + lls
        return distn

    def cond_entropies_for_all_perms(self, return_distn=False):
        r"""
        Returns :math:`H(S_0|Y_k, y^{k-1})` for all possible permutations.

        Dimensions: -1: permutations, -2: y^{k-1}
        :return:
        """
        # distn: -1: y_k, -2: s_k, -3: s_0, -4: permutations, -5: y^{k-1}
        distn = self.distributions_for_all_perms()

        # Marginalize over s_k
        v = distn.logsumexp(-2)

        # Compute :math:`H(S_0, Y_k|y^{k-1})`
        joint_entropies = -(v.exp()*v).sum(-2).sum(-1)

        # Marginalize over s_0
        v = v.logsumexp(-2)

        # Compute :math:`H(Y_k|y^{k-1})`
        dist_entropies = -(v.exp()*v).sum(-1)

        # Compute :math:`H(S_0|Y_k, y^{k-1})`
        cond_entropies = joint_entropies - dist_entropies
        if return_distn:
            return cond_entropies, distn
        return cond_entropies

    def calculate_perm_from_belief(self, return_dict=False):
        r"""Calculates the permutation that minimizes the conditional entropy.
        :math:`\operatorname{argmin}_\sigma H(S_0|Y_k, y^{k-1})`

        :return:
        """
        entropies, distn = self.cond_entropies_for_all_perms(return_distn=True)
        entropies = entropies
        min_entropies, perm_idx = entropies.min(dim=-1)
        perm = self.possible_perms[perm_idx]
        if return_dict:
            return perm, {b'entropy': min_entropies, b'joint_distribution': distn}
        return perm


class ExpandedMinEntropyPolicy(MinEntropyPolicy):

    def __init__(self, possible_perms, skipfirsthmm: SkipFirstDiscreteHMM, trivial_obs=None, save_history=False):
        super(ExpandedMinEntropyPolicy, self).__init__(possible_perms, skipfirsthmm, save_history=save_history)
        self.belief_state = HMMBeliefState.from_skipfirsthmm(skipfirsthmm, trivial_obs=trivial_obs)

    def cond_entropies_for_all_perms(self, return_distn=False):
        """
        Computes the conditional entropy of the initial state given the next step,
        for an hmm that is on an expanded state space.

        Computes :math:`H(L_0|Y_k, y^{k-1})` for all possible permutations.
        :param return_distn:
        :return:
        """
        # distn: -1: y_k, -2: s_k, -3: s_0, -4: permutations, -5: y^{k-1}
        distn = self.distributions_for_all_perms()

        # Marginalize over s_k
        v = distn.logsumexp(-2)

        numout = self.hmm.observation_dist.enumerate_support().shape[0]
        # Reshape the s_0 dimension to be (l_0, o_0)
        # TODO: Refactor this to accept a method for reshaping
        v = v.reshape(v.shape[:-2] + (-1, numout) + v.shape[-1:])

        # Marginalize over o_0
        v = v.logsumexp(-2)

        # Compute :math:`H(L_0, Y_k|y^{k-1})`
        joint_entropies = -(v.exp()*v).sum(-2).sum(-1)

        # Marginalize over l_0
        v = v.logsumexp(-2)

        # Compute :math:`H(Y_k|y^{k-1})`
        dist_entropies = -(v.exp()*v).sum(-1)

        # Compute :math:`H(L_0|Y_k, y^{k-1})`
        cond_entropies = joint_entropies - dist_entropies
        if return_distn:
            return cond_entropies, distn
        return cond_entropies



