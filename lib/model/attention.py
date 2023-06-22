import einops as op
from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
import math
from typing import NamedTuple

from .Config import Config
from .rotary_embedding import rotary_embedding

class Attention(NamedTuple):
    q_proj: Array
    k_proj: Array
    v_proj: Array
    out_proj: Array

@partial(jax.jit, static_argnames=('config',))
def attention(params: Attention, src_seq: Array, dst_seq: Array, attn_mask: Array, *, config: Config) -> Array:
    assert isinstance(params.q_proj, Array)
    assert isinstance(params.k_proj, Array)
    assert isinstance(params.v_proj, Array)

    assert params.q_proj.shape == (config.d_model, config.n_heads, config.d_k)
    assert params.k_proj.shape == (config.d_model, config.n_heads, config.d_k)
    assert params.v_proj.shape == (config.d_model, config.n_heads, config.d_v)
    assert params.out_proj.shape == (config.n_heads, config.d_v, config.d_model)

    assert isinstance(src_seq, Array)
    assert isinstance(dst_seq, Array)
    # assert isinstance(attn_mask, Array)
    # assert attn_mask.dtype == jnp.bool_

    q = op.einsum(src_seq, params.q_proj, 'batch_size src_seq_len d_model, d_model n_heads d_k -> batch_size n_heads src_seq_len d_k')
    k = op.einsum(dst_seq, params.k_proj, 'batch_size dst_seq_len d_model, d_model n_heads d_k -> batch_size n_heads dst_seq_len d_k')
    v = op.einsum(dst_seq, params.v_proj, 'batch_size dst_seq_len d_model, d_model n_heads d_v -> batch_size n_heads dst_seq_len d_v')

    q = rotary_embedding(q)
    k = rotary_embedding(k)

    qk = op.einsum(q, k, 'batch_size n_heads src_seq_len d_k, batch_size n_heads dst_seq_len d_k -> batch_size n_heads src_seq_len dst_seq_len')
    qk /= math.sqrt(config.d_k)

    qkv = op.einsum(qk, v, 'batch_size n_heads src_seq_len dst_seq_len, batch_size n_heads dst_seq_len d_v -> batch_size n_heads src_seq_len d_v')

    out = op.einsum(qkv, params.out_proj, 'batch_size n_heads src_seq_len d_v, n_heads d_v d_model -> batch_size src_seq_len d_model')

    return out
