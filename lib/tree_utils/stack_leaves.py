import jax
import jax.numpy as jnp

# https://docs.liesel-project.org/en/v0.1.4/_modules/liesel/goose/pytree.html#stack_leaves
def stack_leaves(pytrees, axis: int=0):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=axis), *pytrees)
