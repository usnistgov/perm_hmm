import anytree as at


def uniform_tree(num_steps: int, num_outcomes: int):
    """
    Creates a tree of height num_steps, where each internal node has num_outcomes children.

    :return: An AnyTree representing the tree.
    """
    tree = _uniform_tree_helper(num_steps, num_outcomes, at.Node(None))
    return tree


def _uniform_tree_helper(num_steps: int, num_outcomes: int, parent: at.Node):
    """
    Recursively builds a tree of height num_steps, where each internal node has num_outcomes children.

    :return: An AnyTree representing the tree.
    """
    if num_steps == 0:
        return parent
    else:
        for i in range(num_outcomes):
            child = at.Node(i, parent=parent)
            _uniform_tree_helper(num_steps - 1, num_outcomes, child)
        return parent


def list_tree_to_anytree(list_tree, possible_perms):
    r"""Given a tree represented as a list of tensors, makes an
    :py:class:`~anytree.Node` representation of it.

    :param list_tree: The tree, as a list of tensors. The ``i``th tensor
        contains all the nodes at depth ``i`` from the root.
    :param possible_perms: Needed to interpret the edges labelled by
        permutations.
    """
    num_outcomes = list_tree[0].shape[0]
    tree = uniform_tree(len(list_tree), num_outcomes)
    for node in at.LevelOrderIter(tree):
        if node.is_root:
            continue
        address = tuple(n.name for n in node.path[1:])
        p = list_tree[node.depth-1][address]
        node.data = {'perm': possible_perms[p]}
    return tree
