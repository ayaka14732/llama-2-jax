from jax import Array

from .Config import Config

def check_embedding(params: Array, *, config: Config) -> None:
    assert isinstance(params, Array)
    assert params.shape == (config.vocab_size, config.d_model)

def embedding(params: Array, x: Array) -> Array:
    return params[x]
