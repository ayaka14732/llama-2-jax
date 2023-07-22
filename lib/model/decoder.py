from functools import partial
import jax
from jax import Array
import jax.random as rand
from jax.sharding import PositionalSharding

from ..rand_utils import split_key
from .ModelConfig import ModelConfig
from .decoder_block import DecoderBlock, check_decoder_block, create_model_parallel_sharding_decoder_block, decoder_block

Decoder = DecoderBlock

def check_decoder(params: Decoder, *, model_config: ModelConfig) -> None:
    assert isinstance(params, DecoderBlock)
    def inner(state, input_):
        assert isinstance(input_, DecoderBlock)
        check_decoder_block(input_, model_config=model_config)
        return None, None
    jax.lax.scan(inner, None, params)

def create_model_parallel_sharding_decoder(sharding: PositionalSharding) -> Decoder:
    sharding_decoder = create_model_parallel_sharding_decoder_block(sharding)
    return jax.tree_map(lambda x: x.reshape((1, *x.shape)), sharding_decoder)

@partial(jax.jit, static_argnames=('model_config',))
def decoder(params: Decoder, seq: Array, attn_mask: Array, *, key: rand.KeyArray, model_config: ModelConfig) -> Array:
    def inner(state, input_):
        key, seq = state
        key, subkey = split_key(key)
        seq = decoder_block(input_, seq, attn_mask, key=subkey, model_config=model_config)
        return (key, seq), None
    (key, seq), _ = jax.lax.scan(inner, (key, seq), params)
    return seq
