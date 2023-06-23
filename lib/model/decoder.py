from functools import partial
import jax
from jax import Array
from typing import List

from .Config import Config
from .decoder_block import DecoderBlock, decoder_block

Decoder = List[DecoderBlock]

@partial(jax.jit, static_argnames=('config',))
def decoder(params: Decoder, dst_seq: Array, attn_mask: Array, *, config: Config):
    assert isinstance(params, List)
    assert len(params) == config.n_layers

    for param_layer in params:
        dst_seq = decoder_block(param_layer, dst_seq, attn_mask, config=config)
    return dst_seq
