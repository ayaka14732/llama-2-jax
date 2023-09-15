import jax
import jax.numpy as jnp

# https://docs.liesel-project.org/en/v0.1.4/_modules/liesel/goose/pytree.html#stack_leaves
def stack_leaves(pytrees, axis: int=0):
    '''
    Stack the leaves of one or more PyTrees along a new axis.

    Args:
        pytrees: One or more PyTrees.
        axis (int, optional): The axis along which the arrays will be stacked. Default is 0.

    Returns:
        The PyTree with its leaves stacked along the new axis.
    '''
    return jax.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *pytrees)

# https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557
def unstack_leaves(pytrees):
    '''
    Unstack the leaves of a PyTree.

    Args:
        pytrees: A PyTree.

    Returns:
        A list of PyTrees, where each PyTree has the same structure as the input PyTree, but each leaf contains only one part of the original leaf.
    '''
    leaves, treedef = jax.tree_util.tree_flatten(pytrees)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]
