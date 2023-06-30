from jax import Array
from typing import NamedTuple

from .Config import Config
from .llama_model import LlamaModel, check_llama_model

class Llama(NamedTuple):
    model: LlamaModel
    lm_head: Array

def check_llama(params: Llama, *, config: Config) -> None:
    assert isinstance(params.model, LlamaModel)
    assert isinstance(params.lm_head, Array)

    check_llama_model(params.model, config=config)
    assert params.lm_head.shape == (config.d_model, config.vocab_size)
