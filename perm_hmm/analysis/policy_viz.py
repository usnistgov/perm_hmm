"""Tools for visualizing permutation policies.
"""
import os
import argparse
from copy import deepcopy
import torch
import anytree as at
from anytree.exporter import UniqueDotExporter
from perm_hmm.util import id_and_transpositions
from perm_hmm.policies.policy import PermPolicy
from perm_hmm.policies.min_tree import MinEntPolicy
from example_systems.three_states import three_state_hmm
from perm_hmm.analysis.graph_utils import uniform_tree


def list_tree_to_anytree(list_tree, possible_perms):
    num_outcomes = list_tree[0].shape[0]
    tree = uniform_tree(len(list_tree)-1, num_outcomes)
    for node in at.LevelOrderIter(tree):
        if node.is_root:
            continue
        address = tuple(n.name for n in node.path[1:])
        p = list_tree[node.depth-1][address]
        node.data = {'perm': possible_perms[p]}
    return tree


def attach_perms_and_policies_to_tree(tree, policy: PermPolicy):
    """
    At each node in a tree representing data observed, attaches the permutation that would be selected at that point and the policy used up to that point.

    :param tree: An AnyTree representing the data.
    :param policy: A PermPolicy object.
    :return: The same tree, but with the permutation and the policy used up to that point attached to each node.
    """
    node = tree.root
    s = deepcopy(policy)
    s.reset()
    node.data = {'policy': s}
    for node in at.LevelOrderIter(tree):
        if node.is_root:
            continue
        s = deepcopy(node.parent.data['policy'])
        d = torch.tensor([node.name])
        p = s.get_perm(d)
        node.data = {'policy': s, 'perm': p}
    return tree


def remove_policies(tree):
    """
    Given a tree with policies in dictionaries attached to each node, removes the policies.

    :param tree: An AnyTree representing the data.
    :return: The same tree, but with the policies removed.
    """
    for node in at.PreOrderIter(tree):
        del node.data['policy']
    return tree


def attach_perms_to_tree(tree, policy: PermPolicy):
    """
    At each node in a tree representing data observed, attaches the permutation that would be selected at that point.

    :param tree:
    :param policy:
    :return:
    """
    tree = attach_perms_and_policies_to_tree(tree, policy)
    tree = remove_policies(tree)
    return tree


def make_full_decision_tree(policy, num_steps):
    num_outcomes = policy.hmm.enumerate_support().shape[0]
    tree = uniform_tree(num_steps, num_outcomes)
    tree = attach_perms_to_tree(tree, policy)
    return tree


def main(minus_log_a=3, minus_log_b=3, num_steps=3, output_file=None, save_graph=False):
    if output_file is None:
        output_file = 'policy_viz.dot'
        os.path.join(os.getcwd(), output_file)
    if not save_graph:
        directory = os.path.dirname(output_file)
        directory = os.path.join(directory, '../../example_scripts/graphs')
        if not os.path.exists(directory):
            os.makedirs(directory)
        output_file = os.path.join(directory, output_file)
    hmm = three_state_hmm(-minus_log_a, -minus_log_b)
    possible_perms = id_and_transpositions(hmm.initial_logits.shape[0])
    policy = MinEntPolicy(possible_perms, hmm)
    tree = make_full_decision_tree(policy, num_steps)

    def nodeattrfunc(node):
        if node.is_root:
            return '"label"="root"'
        return '"label"="{}"'.format(str(list(node.data['perm'][0].numpy())))

    def edgeattrfunc(node, child):
        return '"label"="{}"'.format(child.name)

    exporter = UniqueDotExporter(tree.root, edgeattrfunc=edgeattrfunc, nodeattrfunc=nodeattrfunc)
    exporter.to_dotfile(output_file)
    # if not save_graph:
    #     os.remove(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--minus_log_a', type=int, default=3)
    parser.add_argument('--minus_log_b', type=int, default=3)
    parser.add_argument('--num_steps', type=int, default=3)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--save_graph', action='store_true', default=False)
    args = parser.parse_args()
    minus_log_a = args.minus_log_a
    minus_log_b = args.minus_log_b
    num_steps = args.num_steps
    output_file = args.output_file
    save_graph = args.save_graph
    main(minus_log_a, minus_log_b, num_steps, output_file, save_graph)


