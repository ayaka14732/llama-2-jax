from functools import partial
import jax
from jax import Array
import jax.random as rand
from typing import NamedTuple

from .attention import Attention, attention, check_attention
from .Config import Config
from .dropout import dropout
from .rms_norm import check_rms_norm, rms_norm

class DecoderBlock(NamedTuple):
    input_norm: Array
    attention: Attention
    post_attn_norm: Array
    gate_proj: Array
    up_proj: Array
    down_proj: Array

def check_decoder_block(params: DecoderBlock, *, config: Config) -> None:
    assert isinstance(params.input_norm, Array)
    assert isinstance(params.attention, Attention)
    assert isinstance(params.post_attn_norm, Array)
    assert isinstance(params.gate_proj, Array)
    assert isinstance(params.up_proj, Array)
    assert isinstance(params.down_proj, Array)

    check_rms_norm(params.input_norm, config=config)
    check_attention(params.attention, config=config)
    check_rms_norm(params.post_attn_norm, config=config)
    assert params.gate_proj.shape == (config.d_model, config.d_ff)
    assert params.up_proj.shape == (config.d_model, config.d_ff)
    assert params.down_proj.shape == (config.d_ff, config.d_model)

@partial(jax.jit, static_argnames=('config',))
def decoder_block(params: DecoderBlock, seq: Array, attn_mask: Array, *, key: rand.KeyArray, config: Config) -> Array:
    if config.dropout_rate is None:
        key0 = key1 = key2 = None
    else:
        key0, key1, key2 = rand.split(key, num=3)

    seq_ = seq
    seq = rms_norm(params.input_norm, seq, config=config)
    seq = attention(params.attention, seq, seq, attn_mask, config=config)
    seq = dropout(seq, key=key0, config=config)
    seq += seq_

    seq_ = seq
    seq = rms_norm(params.post_attn_norm, seq, config=config)
    ff = jax.nn.silu(seq @ params.gate_proj) * (seq @ params.up_proj)
    ff = dropout(ff, key=key1, config=config)
    seq = ff @ params.down_proj
    seq = dropout(seq, key=key2, config=config)
    seq += seq_

    return seq
