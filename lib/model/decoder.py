from functools import partial
import jax
from jax import Array
import jax.random as rand

from .ModelConfig import ModelConfig
from .decoder_block import DecoderBlock, check_decoder_block, decoder_block

Decoder = list[DecoderBlock]

def check_decoder(params: Decoder, *, model_config: ModelConfig) -> None:
    assert len(params) == model_config.n_layers
    for params_layer in params:
        assert isinstance(params_layer, DecoderBlock)
        check_decoder_block(params_layer, model_config=model_config)

@partial(jax.jit, static_argnames=('model_config',))
def decoder(params: Decoder, dst_seq: Array, attn_mask: Array, *, key: rand.KeyArray, model_config: ModelConfig) -> Array:
    for params_layer in params:
        dst_seq = decoder_block(params_layer, dst_seq, attn_mask, key=key, model_config=model_config)
    return dst_seq
