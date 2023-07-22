from functools import partial
import jax
from jax import Array
import jax.random as rand
from jax.sharding import PositionalSharding
from typing import NamedTuple

from ..rand_utils import split_key
from .attention import Attention, attention, check_attention, create_model_parallel_sharding_attention
from .ModelConfig import ModelConfig
from .dropout import dropout
from .rms_norm import check_rms_norm, rms_norm

class DecoderBlock(NamedTuple):
    input_norm: Array
    attention: Attention
    post_attn_norm: Array
    gate_proj: Array
    up_proj: Array
    down_proj: Array

def check_decoder_block(params: DecoderBlock, *, model_config: ModelConfig) -> None:
    assert isinstance(params.input_norm, Array)
    assert isinstance(params.attention, Attention)
    assert isinstance(params.post_attn_norm, Array)
    assert isinstance(params.gate_proj, Array)
    assert isinstance(params.up_proj, Array)
    assert isinstance(params.down_proj, Array)

    check_rms_norm(params.input_norm, model_config=model_config)
    check_attention(params.attention, model_config=model_config)
    check_rms_norm(params.post_attn_norm, model_config=model_config)
    assert params.gate_proj.shape == (model_config.d_model, model_config.d_ff)
    assert params.up_proj.shape == (model_config.d_model, model_config.d_ff)
    assert params.down_proj.shape == (model_config.d_ff, model_config.d_model)

def create_model_parallel_sharding_decoder_block(sharding: PositionalSharding) -> DecoderBlock:
    input_norm = sharding.replicate((0,))
    attention = create_model_parallel_sharding_attention(sharding)
    post_attn_norm = sharding.replicate((0,))
    gate_proj = sharding.reshape((1, -1))
    up_proj = sharding.reshape((1, -1))
    down_proj = sharding.reshape((-1, 1))
    return DecoderBlock(input_norm, attention, post_attn_norm, gate_proj, up_proj, down_proj)

@partial(jax.jit, static_argnames=('model_config',))
def decoder_block(params: DecoderBlock, seq: Array, attn_mask: Array, *, key: rand.KeyArray, model_config: ModelConfig) -> Array:
    key0, key1, key2 = split_key(key, num=3)

    seq_ = seq
    seq = rms_norm(params.input_norm, seq, model_config=model_config)
    seq = attention(params.attention, seq, seq, attn_mask, model_config=model_config)
    seq = dropout(seq, key=key0, model_config=model_config)
    seq += seq_

    seq_ = seq
    seq = rms_norm(params.post_attn_norm, seq, model_config=model_config)
    ff = jax.nn.silu(seq @ params.gate_proj) * (seq @ params.up_proj)
    ff = dropout(ff, key=key1, model_config=model_config)
    seq = ff @ params.down_proj
    seq = dropout(seq, key=key2, model_config=model_config)
    seq += seq_

    return seq
