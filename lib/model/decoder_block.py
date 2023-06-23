from functools import partial
import jax
from jax import Array
from typing import NamedTuple

from .attention import attention
from .Config import Config
from .rms_norm import rms_norm

class DecoderBlock(NamedTuple):
    input_norm: Array
    attention: Array
    post_attn_norm: Array
    gate_proj: Array
    up_proj: Array
    down_proj: Array

@partial(jax.jit, static_argnames=('config',))
def decoder_block(params: DecoderBlock, seq: Array, attn_mask: Array, *, config: Config) -> Array:
    assert isinstance(seq, Array)
    # assert isinstance(attn_mask, Array)
    assert params.up_proj.shape == (config.d_model, config.d_ff)
    assert params.down_proj.shape == (config.d_ff, config.d_model)

    seq_ = seq
    seq = rms_norm(params.input_norm, seq, config=config)
    seq = attention(params.attention, seq, seq, attn_mask, config=config)
    seq += seq_

    seq_ = seq
    seq = rms_norm(params.post_attn_norm, seq, config=config)
    ff = jax.nn.silu(seq @ params.gate_proj) * (seq @ params.up_proj)
    seq = ff @ params.down_proj
    seq += seq_

    return seq
