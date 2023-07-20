from functools import partial
import jax
from jax import Array
import jax.numpy as jnp

from .ModelConfig import ModelConfig

def check_rms_norm(params: Array, *, model_config: ModelConfig) -> None:
    assert isinstance(params, Array)
    assert params.shape == (model_config.d_model,)

def shard_rms_norm(params: Array) -> Array:
    from jax.sharding import PositionalSharding; devices = jax.devices(); shards = PositionalSharding(devices); n_shard = len(devices)
    return jax.device_put(params, shards.replicate((0,)))

# Taken from https://github.com/ztjhz/t5-jax/blob/main/model/layer_norm.py#L23
@partial(jax.jit, static_argnames=('model_config',))
def rms_norm(params: Array, x: Array, *, model_config: ModelConfig) -> Array:
    x_rms = jnp.sqrt((x * x).mean(axis=-1, keepdims=True) + model_config.rms_norm_eps)
    y = x / x_rms * params
    return y
