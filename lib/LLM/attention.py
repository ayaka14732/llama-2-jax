from functools import partial
import math
from typing import Any, NamedTuple

import einops as op
import jax
from jax import Array
import jax.nn as nn
import jax.numpy as jnp
import jax.random as rand

from .ModelConfig import ModelConfig
from .kv_cache import KVCache
from .rotary_embedding import RotaryValues, forward_rotary_embedding

class Attention(NamedTuple):
    q_proj: Any  # Array
    k_proj: Any  # Array
    v_proj: Any  # Array
    out_proj: Any  # Array

def check_attention(params: Attention, *, model_config: ModelConfig) -> None:
    assert isinstance(params.q_proj, Array)
    assert isinstance(params.k_proj, Array)
    assert isinstance(params.v_proj, Array)
    assert isinstance(params.out_proj, Array)

    assert params.q_proj.shape == (model_config.d_model, model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_k)
    assert params.k_proj.shape == (model_config.d_model, model_config.n_heads_kv, model_config.d_k)
    assert params.v_proj.shape == (model_config.d_model, model_config.n_heads_kv, model_config.d_v)
    assert params.out_proj.shape == (model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_v, model_config.d_model)

def init_attention(*, key: Array, model_config: ModelConfig) -> Attention:
    upper = 1. / math.sqrt(model_config.d_model)
    key0, key1, key2, key3 = rand.split(key, num=4)
    q_proj = rand.truncated_normal(key0, -upper, upper, (model_config.d_model, model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_k))
    k_proj = rand.truncated_normal(key1, -upper, upper, (model_config.d_model, model_config.n_heads_kv, model_config.d_k))
    v_proj = rand.truncated_normal(key2, -upper, upper, (model_config.d_model, model_config.n_heads_kv, model_config.d_v))
    out_proj = rand.truncated_normal(key3, -upper, upper, (model_config.n_rep_kv, model_config.n_heads_kv, model_config.d_v, model_config.d_model))
    return Attention(q_proj, k_proj, v_proj, out_proj)

@partial(jax.jit, static_argnames=('model_config',))
def forward_attention(params: Attention, src_seq: Array, dst_seq: Array, qk_mask: Array, *, rotary_values: RotaryValues, kv_cache: KVCache | None=None, model_config: ModelConfig) -> tuple[Array, KVCache | None]:
    q = op.einsum(src_seq, params.q_proj, 'B S M, M R H K -> B R H S K')
    k = op.einsum(dst_seq, params.k_proj, 'B D M, M H K -> B H D K')
    v = op.einsum(dst_seq, params.v_proj, 'B D M, M H V -> B H D V')

    q = forward_rotary_embedding(q, rotary_values=rotary_values)
    k = forward_rotary_embedding(k, rotary_values=rotary_values)

    if kv_cache is not None:
        assert src_seq.shape[1] == 1
        assert dst_seq.shape[1] == 1
        k_cache, v_cache = kv_cache
        k = k_cache.at[:, :, -1:].set(k)
        v = v_cache.at[:, :, -1:].set(v)

    qk = op.einsum(q, k, 'B R H S K, B H D K -> B R H S D')
    qk /= math.sqrt(model_config.d_k)
    qk = jnp.where(qk_mask, qk, -jnp.inf)
    qk = nn.softmax(qk)  # TODO: use `where`
    # qk = nn.softmax(qk, where=qk_mask, initial=0.)
    qk = jnp.where(qk_mask, qk, 0)  # TODO: why this line?

    qkv = op.einsum(qk, v, 'B R H S D, B H D V -> B R H S V')
    out = op.einsum(qkv, params.out_proj, 'B R H S V, R H V M -> B S M')
    kv_cache = None if not model_config.return_kv_cache else KVCache(k, v)

    return out, kv_cache
