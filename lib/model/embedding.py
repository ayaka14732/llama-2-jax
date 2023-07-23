from jax import Array

from .ModelConfig import ModelConfig

def check_embedding(params: Array, *, model_config: ModelConfig) -> None:
    assert isinstance(params, Array)
    assert params.shape == (model_config.vocab_size, model_config.d_model)

def embedding(params: Array, x: Array) -> Array:
    return params[x]
