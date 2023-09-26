from functools import partial
import jax
from jax import Array
import jax.random as rand
import math
from typing import Any, NamedTuple

from .ModelConfig import ModelConfig
from .attention import KVCache
from .llama_model import LlamaModel, check_llama_model, forward_llama_model, init_llama_model

class Llama(NamedTuple):
    model: LlamaModel
    lm_head: Any  # Array

def check_llama(params: Llama, *, model_config: ModelConfig) -> None:
    assert isinstance(params.model, LlamaModel)
    assert isinstance(params.lm_head, Array)

    check_llama_model(params.model, model_config=model_config)
    assert params.lm_head.shape == (model_config.d_model, model_config.vocab_size)

def init_llama(*, key: Array, model_config: ModelConfig) -> Llama:
    upper = 1. / math.sqrt(model_config.d_model)
    key0, key1 = rand.split(key)
    model = init_llama_model(key=key0, model_config=model_config)
    lm_head = rand.truncated_normal(key1, -upper, upper, (model_config.d_model, model_config.vocab_size))
    return Llama(model, lm_head)

@partial(jax.jit, static_argnames=('model_config'))
def forward_llama(params: Llama, seq: Array, attn_mask: Array, *, cache_position: Array | None=None, kv_cache: KVCache | None=None, key: Array | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    outputs, kv_cache = forward_llama_model(params.model, seq, attn_mask, cache_position=cache_position, kv_cache=kv_cache, key=key, model_config=model_config)
    logits = outputs @ params.lm_head
    return logits, kv_cache
