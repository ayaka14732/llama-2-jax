import jax.numpy as jnp
from typing import Any, NamedTuple

from .ModelConfig import ModelConfig

class KVCache(NamedTuple):
    k_cache: Any  # Array
    v_cache: Any  # Array

def init_kv_cache(batch_size: int, dst_len: int, *, model_config: ModelConfig) -> KVCache:
    k_cache = jnp.zeros((model_config.n_layers, batch_size, model_config.n_heads_kv, dst_len, model_config.d_k))
    v_cache = jnp.zeros((model_config.n_layers, batch_size, model_config.n_heads_kv, dst_len, model_config.d_v))
    return KVCache(k_cache, v_cache)
