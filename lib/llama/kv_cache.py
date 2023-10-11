from typing import Any, NamedTuple

import jax.numpy as jnp

class KVCache(NamedTuple):
    k_cache: Any  # Array
    v_cache: Any  # Array

def shift_left_kv_cache(kv_cache: KVCache) -> KVCache:
    k_cache, v_cache = kv_cache
    k_cache = jnp.roll(k_cache, -1, axis=-2)  # -2: dimension L
    v_cache = jnp.roll(v_cache, -1, axis=-2)  # -2: dimension L
    return KVCache(k_cache, v_cache)
