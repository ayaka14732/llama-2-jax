import einops as op
from functools import partial
import jax
from jax import Array
import jax.nn as nn
import jax.numpy as jnp
import jax.random as rand
import math
from typing import Any, NamedTuple

from .ModelConfig import ModelConfig
from .rotary_embedding import forward_rotary_embedding

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
def forward_attention(params: Attention, src_seq: Array, dst_seq: Array, attn_mask: Array, *, model_config: ModelConfig) -> Array:
    q = op.einsum(src_seq, params.q_proj, 'batch_size src_seq_len d_model, d_model n_rep_kv n_heads_kv d_k -> batch_size n_rep_kv n_heads_kv src_seq_len d_k')
    k = op.einsum(dst_seq, params.k_proj, 'batch_size dst_seq_len d_model, d_model n_heads_kv d_k -> batch_size n_heads_kv dst_seq_len d_k')
    v = op.einsum(dst_seq, params.v_proj, 'batch_size dst_seq_len d_model, d_model n_heads_kv d_v -> batch_size n_heads_kv dst_seq_len d_v')

    q = forward_rotary_embedding(q)
    k = forward_rotary_embedding(k)

    qk = op.einsum(q, k, 'batch_size n_rep_kv n_heads_kv src_seq_len d_k, batch_size n_heads_kv dst_seq_len d_k -> batch_size n_rep_kv n_heads_kv src_seq_len dst_seq_len')
    qk /= math.sqrt(model_config.d_k)
    qk = jnp.where(attn_mask, qk, -jnp.inf)
    qk = nn.softmax(qk)
    qk = jnp.where(attn_mask, qk, 0)  # TODO: why this line?

    qkv = op.einsum(qk, v, 'batch_size n_rep_kv n_heads_kv src_seq_len dst_seq_len, batch_size n_heads_kv dst_seq_len d_v -> batch_size n_rep_kv n_heads_kv src_seq_len d_v')

    out = op.einsum(qkv, params.out_proj, 'batch_size n_rep_kv n_heads_kv src_seq_len d_v, n_rep_kv n_heads_kv d_v d_model -> batch_size src_seq_len d_model')
    return out
