import math

from jax import Array
import jax.random as rand

from .ModelConfig import ModelConfig

def check_embedding(params: Array, *, model_config: ModelConfig) -> None:
    assert isinstance(params, Array)
    assert params.shape == (model_config.vocab_size, model_config.d_model)

def init_embedding(*, key: Array, model_config: ModelConfig) -> Array:
    upper = 1. / math.sqrt(model_config.d_model)
    return rand.truncated_normal(key, -upper, upper, (model_config.vocab_size, model_config.d_model))

def forward_embedding(params: Array, x: Array) -> Array:
    return params[x]
