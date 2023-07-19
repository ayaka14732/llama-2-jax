from jax import Array
from typing import NamedTuple

from .ModelConfig import ModelConfig
from .llama_model import LlamaModel, check_llama_model

class Llama(NamedTuple):
    model: LlamaModel
    lm_head: Array

def check_llama(params: Llama, *, model_config: ModelConfig) -> None:
    assert isinstance(params.model, LlamaModel)
    assert isinstance(params.lm_head, Array)

    check_llama_model(params.model, model_config=model_config)
    assert params.lm_head.shape == (model_config.d_model, model_config.vocab_size)
