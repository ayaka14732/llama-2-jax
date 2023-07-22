from jax import Array
from jax.sharding import PositionalSharding
from typing import NamedTuple

from .ModelConfig import ModelConfig
from .llama_model import LlamaModel, check_llama_model, create_model_parallel_sharding_llama_model

class Llama(NamedTuple):
    model: LlamaModel
    lm_head: Array

def check_llama(params: Llama, *, model_config: ModelConfig) -> None:
    assert isinstance(params.model, LlamaModel)
    assert isinstance(params.lm_head, Array)

    check_llama_model(params.model, model_config=model_config)
    assert params.lm_head.shape == (model_config.d_model, model_config.vocab_size)

def create_model_parallel_sharding_llama(sharding: PositionalSharding) -> Llama:
    model = create_model_parallel_sharding_llama_model(sharding)
    lm_head = sharding.replicate((0,))
    return Llama(model, lm_head)
