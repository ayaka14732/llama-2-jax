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
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *pytrees)

# https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557
def unstack_leaves(pytrees):
    '''
    Unstack the leaves of a PyTree.

    Args:
        pytrees: A PyTree.

    Returns:
        A list of PyTrees, where each PyTree has the same structure as the input PyTree, but with only one of the original leaves.
    '''
    leaves, treedef = jax.tree_util.tree_flatten(pytrees)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]

def tree_apply(func, *pytrees):
    '''
    Apply a function to the leaves of one or more PyTrees.

    Args:
        func (callable): Function to apply to each leaf. It must take the same number of arguments as there are PyTrees.
        pytrees: One or more PyTrees.

    Returns:
        A new PyTree with the same structure as the input PyTrees, but with the function applied to each leaf.

    Example:
        >>> tree_apply(lambda a, b: a + b + 1, [[[1, 2]]], [[[3, 4]]])
        [[[5, 7]]]
    '''
    leaves, treedefs = zip(*([jax.tree_util.tree_flatten(pytree) for pytree in pytrees]), strict=True)
    treedef = treedefs[0]
    results = [func(*args) for args in zip(*leaves, strict=True)]
    return treedef.unflatten(results)
