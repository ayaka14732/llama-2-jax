from functools import partial

import jax
from jax import Array
import jax.random as rand

from .ModelConfig import ModelConfig

@partial(jax.jit, static_argnames=('model_config',))
def forward_dropout(x: Array, *, key: Array | None=None, model_config: ModelConfig) -> Array:
    if key is None or model_config.dropout_rate is None:  # should disable dropout
        return x

    assert 0. <= model_config.dropout_rate <= 1.
    assert isinstance(x, Array)
    assert isinstance(key, Array)

    keep_rate = 1. - model_config.dropout_rate
    out = x * rand.bernoulli(key, p=keep_rate, shape=x.shape) / keep_rate
    assert x.shape == out.shape
    return out
