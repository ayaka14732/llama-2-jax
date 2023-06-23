from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
from typing import NamedTuple

from .Config import Config
from .decoder import Decoder, decoder
from .embedding import embedding
from .rms_norm import rms_norm

class Llama(NamedTuple):
    embedding: Array
    decoder: Decoder
    norm: Array

@partial(jax.jit, static_argnames=('config',))
def llama(params: Llama, seq: Array, attn_mask: Array, *, config=Config):
    assert isinstance(attn_mask, Array)
    assert seq.shape == attn_mask.shape
    assert attn_mask.dtype == jnp.bool_

    attn_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask))[:, None]

    seq = embedding(params, seq, config=config)
    seq = decoder(params, seq, attn_mask, config=config)
    seq = rms_norm(params, seq, config=config)
    return seq
