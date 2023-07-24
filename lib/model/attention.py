import einops as op
from functools import partial
import jax
from jax import Array
import jax.nn as nn
import jax.numpy as jnp
from jax.sharding import PositionalSharding
import math
from typing import Any, NamedTuple

from .ModelConfig import ModelConfig
from .rotary_embedding import rotary_embedding

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

@partial(jax.jit, static_argnames=('model_config',))
def attention(params: Attention, src_seq: Array, dst_seq: Array, attn_mask: Array, *, model_config: ModelConfig) -> Array:
    q = op.einsum(src_seq, params.q_proj, 'batch_size src_seq_len d_model, d_model n_rep_kv n_heads_kv d_k -> batch_size n_rep_kv n_heads_kv src_seq_len d_k')
    k = op.einsum(dst_seq, params.k_proj, 'batch_size dst_seq_len d_model, d_model n_heads_kv d_k -> batch_size n_heads_kv dst_seq_len d_k')
    v = op.einsum(dst_seq, params.v_proj, 'batch_size dst_seq_len d_model, d_model n_heads_kv d_v -> batch_size n_heads_kv dst_seq_len d_v')

    q = rotary_embedding(q)
    k = rotary_embedding(k)

    qk = op.einsum(q, k, 'batch_size n_rep_kv n_heads_kv src_seq_len d_k, batch_size n_heads_kv dst_seq_len d_k -> batch_size n_rep_kv n_heads_kv src_seq_len dst_seq_len')
    qk /= math.sqrt(model_config.d_k)
    qk = jnp.where(attn_mask, qk, jnp.NINF)
    qk = nn.softmax(qk)
    qk = jnp.where(attn_mask, qk, 0)

    qkv = op.einsum(qk, v, 'batch_size n_rep_kv n_heads_kv src_seq_len dst_seq_len, batch_size n_heads_kv dst_seq_len d_v -> batch_size n_rep_kv n_heads_kv src_seq_len d_v')

    out = op.einsum(qkv, params.out_proj, 'batch_size n_rep_kv n_heads_kv src_seq_len d_v, n_rep_kv n_heads_kv d_v d_model -> batch_size src_seq_len d_model')
    return out
