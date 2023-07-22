import jax
from jax import Array
from jax.sharding import PositionalSharding

from .ModelConfig import ModelConfig

def check_embedding(params: Array, *, model_config: ModelConfig) -> None:
    assert isinstance(params, Array)
    assert params.shape == (model_config.vocab_size, model_config.d_model)

def create_model_parallel_sharding_embedding(sharding: PositionalSharding) -> PositionalSharding:
    return sharding.replicate((0,))

def embedding(params: Array, x: Array) -> Array:
    return params[x]
