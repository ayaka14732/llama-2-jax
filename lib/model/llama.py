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
def llama(params: Llama, seq: Array, mask: Array, *, config=Config):
    assert mask.dtype == jnp.bool_

    seq = embedding(params, seq, config=config)
    seq = decoder(params, seq, mask, config=config)
    seq = rms_norm(params, seq, config=config)
    return seq
