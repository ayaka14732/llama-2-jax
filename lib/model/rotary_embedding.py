import einops as op
from functools import partial
import jax
import jax.numpy as jnp

from .Config import Config

# TODO: Mostly taken from https://github.com/kingoflolz/mesh-transformer-jax/blob/master/mesh_transformer/layers.py
def fixed_pos_embedding(seq_len, d_kv):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, d_kv, 2) / d_kv))
    sinusoid_inp = jnp.einsum('i , j -> i j', jnp.arange(seq_len), inv_freq)
    sin_val = jnp.sin(sinusoid_inp)
    cos_val = jnp.cos(sinusoid_inp)
    sin_val = op.repeat(sin_val, 'b n -> b (n j)', j=2)
    cos_val = op.repeat(cos_val, 'b n -> b (n j)', j=2)
    return sin_val, cos_val

def rotate_every_two(x):
    x = op.rearrange(x, '... (x i) -> ... x i', i=2)
    x = x[..., ::-1] * jnp.array([-1, 1])
    x = op.rearrange(x, '... x i -> ... (x i)')
    return x

def rotary_embedding(x):
    # assert config.d_k % 2 == 0
    # assert config.d_v % 2 == 0

    seq_len = x.shape[-2]
    d_kv = x.shape[-1]  # flexible variable that can be either `d_k` or `d_v` depending on the context

    with jax.ensure_compile_time_eval():
        sin_val, cos_val = fixed_pos_embedding(seq_len, d_kv)

    pattern = 'batch_size n_heads seq_len d_kv, seq_len d_kv -> batch_size n_heads seq_len d_kv'
    a = op.einsum(x, cos_val, pattern)
    b = op.einsum(rotate_every_two(x), sin_val, pattern)
    return a + b
