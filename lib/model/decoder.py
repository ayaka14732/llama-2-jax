from functools import partial
import jax
from jax import Array
import jax.random as rand

from .Config import Config
from .decoder_block import DecoderBlock, check_decoder_block, decoder_block

Decoder = list[DecoderBlock]

def check_decoder(params: Decoder, *, config: Config) -> None:
    assert len(params) == config.n_layers
    for params_layer in params:
        assert isinstance(params_layer, DecoderBlock)
        check_decoder_block(params_layer, config=config)

@partial(jax.jit, static_argnames=('config',))
def decoder(params: Decoder, dst_seq: Array, attn_mask: Array, *, key: rand.KeyArray, config: Config) -> Array:
    for params_layer in params:
        dst_seq = decoder_block(params_layer, dst_seq, attn_mask, key=key, config=config)
    return dst_seq
