from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
from typing import NamedTuple

from .Config import Config
from .decoder import Decoder, check_decoder, decoder
from .embedding import check_embedding, embedding
from .rms_norm import check_rms_norm, rms_norm

class Llama(NamedTuple):
    embedding: Array
    decoder: Decoder
    norm: Array

def check_llama(params: Llama, *, config: Config) -> None:
    assert isinstance(params.embedding, Llama)
    assert isinstance(params.decoder, Decoder)
    assert isinstance(params.norm, Array)

    check_embedding(params.embedding, config=config)
    check_decoder(params.decoder, config=config)
    check_rms_norm(params.norm, config=config)

@partial(jax.jit, static_argnames=('config',))
def llama(params: Llama, seq: Array, attn_mask: Array, *, config=Config) -> Array:
    # assert isinstance(attn_mask, Array)
    # assert seq.shape == attn_mask.shape
    # assert attn_mask.dtype == jnp.bool_

    # assert isinstance(src_seq, Array)
    # assert isinstance(dst_seq, Array)
    # assert isinstance(attn_mask, Array)
    # assert attn_mask.dtype == jnp.bool_
    # assert isinstance(seq, Array)
    assert seq.dtype == jnp.uint16
    # assert d_k % 2 == 0

    attn_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask))[:, None]

    seq = embedding(params, seq)
    seq = decoder(params, seq, attn_mask, config=config)
    seq = rms_norm(params, seq, config=config)
    return seq
