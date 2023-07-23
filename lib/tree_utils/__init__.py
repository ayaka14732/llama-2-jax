import jax
import jax.numpy as jnp

# https://docs.liesel-project.org/en/v0.1.4/_modules/liesel/goose/pytree.html#stack_leaves
def stack_leaves(pytrees, axis: int=0):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *pytrees)

# https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75?permalink_comment_id=4634557#gistcomment-4634557
def unstack_leaves(pytrees):
    leaves, treedef = jax.tree_util.tree_flatten(pytrees)
    return [treedef.unflatten(leaf) for leaf in zip(*leaves, strict=True)]

from .tree_apply import tree_apply
