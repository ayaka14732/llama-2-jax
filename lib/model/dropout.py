from functools import partial
import jax
from jax import Array
import jax.random as rand

from .Config import Config

@partial(jax.jit, static_argnames=('config',))
def dropout(x: Array, *, key: rand.KeyArray, config: Config) -> Array:
    if config.dropout_rate is None:  # disable dropout
        return x

    assert 0. <= config.dropout_rate <= 1.
    assert isinstance(x, Array)
    assert isinstance(key, rand.KeyArray)

    keep_rate = 1. - config.dropout_rate
    out = x * rand.bernoulli(key, p=keep_rate, shape=x.shape) / keep_rate
    assert x.shape == out.shape
    return out
