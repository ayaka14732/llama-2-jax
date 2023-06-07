from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand

from .Config import Config

@partial(jax.jit, static_argnames=('config',))
def dropout(params: Array, *, key: rand.KeyArray, config: Config) -> jnp.array:
    dropout_rate = config['dropout_rate']
    assert 0. <= dropout_rate <= 1.
    assert isinstance(params, Array)
    assert isinstance(key, rand.KeyArray)

    keep_rate = 1. - dropout_rate
    out = params * rand.bernoulli(key, p=keep_rate, shape=params.shape) / keep_rate
    assert params.shape == out.shape
    return out
