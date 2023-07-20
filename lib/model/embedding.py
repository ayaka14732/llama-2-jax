import jax
from jax import Array

from .ModelConfig import ModelConfig

def check_embedding(params: Array, *, model_config: ModelConfig) -> None:
    assert isinstance(params, Array)
    assert params.shape == (model_config.vocab_size, model_config.d_model)

def shard_embedding(params: Array) -> Array:
    from jax.sharding import PositionalSharding; devices = jax.devices(); shards = PositionalSharding(devices); n_shard = len(devices)
    return jax.device_put(params, shards.replicate((0,)))

def embedding(params: Array, x: Array) -> Array:
    return params[x]
