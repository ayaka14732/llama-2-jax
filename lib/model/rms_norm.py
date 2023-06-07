from functools import partial
import jax
from jax import Array
import jax.numpy as jnp

from .Config import Config

# Taken from  https://github.com/ztjhz/t5-jax/blob/main/model/layer_norm.py#L23
@partial(jax.jit, static_argnames=('config',))
def rms_norm(params: Array, x: Array, eps: float=1e-6, *, config: Config) -> jnp.ndarray:
    assert isinstance(params, Array)
    assert isinstance(x, Array)

    x_rms = jnp.sqrt((x * x).mean(axis=-1, keepdims=True) + eps)
    y = x / x_rms * params

    return y
