from functools import partial
import jax
from jax import Array
from typing import List

from .Config import Config
from .decoder_block import DecoderBlock, decoder_block

DecoderBlocks = List[DecoderBlock]

@partial(jax.jit, static_argnames=('config',))
def decoder_blocks(params: DecoderBlocks, dst_seq: Array, *, config: Config):
    assert isinstance(params, List)
    assert len(params) == config['n_layers']
    assert isinstance(dst_seq, Array)
    assert dst_seq.shape[-1] == config['d_model']

    for param_layer in params:
        dst_seq = decoder_block(param_layer, dst_seq)
    return dst_seq
