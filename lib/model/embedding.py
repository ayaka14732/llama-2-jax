from functools import partial
import jax
from jax import Array
import jax.numpy as jnp

from .Config import Config

@partial(jax.jit, static_argnames=('config',))
def embedding(params: Array, x: Array, *, config: Config) -> Array:
    assert isinstance(params, Array)
    assert params.dtype == jnp.uint16
    assert params.shape == (config['vocab_size'], config['d_model'])

    y = params[x]
    return y
