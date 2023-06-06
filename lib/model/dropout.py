from functools import partial
import jax
import jax.numpy as jnp
import jax.random as rand

@partial(jax.jit, static_argnames=('keep_rate',))
def dropout(params: jnp.array, key: rand.KeyArray, *, keep_rate: float=0.9) -> jnp.array:
    assert 0. <= keep_rate <= 1.

    out = params * rand.bernoulli(key, p=keep_rate, shape=params.shape) / keep_rate
    assert params.shape == out.shape
    return out
