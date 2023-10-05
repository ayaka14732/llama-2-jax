import einops as op
import jax
from jax import Array
import jax.numpy as jnp

# TODO: Mostly taken from https://github.com/kingoflolz/mesh-transformer-jax/blob/master/mesh_transformer/layers.py
# and https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L92
def _make_weights(seq_len: int, d_k: int) -> tuple[Array, Array]:
    inv_freq = 1. / (10000 ** (jnp.arange(0, d_k, 2) / d_k))
    sinusoid_inp = op.einsum(jnp.arange(seq_len), inv_freq, 'seq_len, j -> seq_len j')
    sin_val = jnp.sin(sinusoid_inp)
    cos_val = jnp.cos(sinusoid_inp)
    sin_val = op.repeat(sin_val, 'seq_len d_k -> seq_len (i d_k)', i=2)
    cos_val = op.repeat(cos_val, 'seq_len d_k -> seq_len (i d_k)', i=2)
    return sin_val, cos_val

def _rotate_half(x: Array) -> Array:
    x = op.rearrange(x, '... (i x) -> ... i x', i=2)  # split the last dimension: (..., n) -> (..., 2, n // 2)
    x = x[..., ::-1, :]  # reverse dimension -2
    x = x.at[..., 0, :].multiply(-1)  # negate the first half of dimension -2
    x = op.rearrange(x, '... i x -> ... (i x)')  # merge the last two dimensions: (..., 2, n // 2) -> (..., n)
    return x

def forward_rotary_embedding(m: Array) -> Array:
    *_, seq_len, d_k = m.shape

    with jax.ensure_compile_time_eval():
        sin_val, cos_val = _make_weights(seq_len, d_k)
        assert sin_val.dtype == jnp.float32
        assert cos_val.dtype == jnp.float32

    n = _rotate_half(m)
    a = op.einsum(m, cos_val, '... seq_len d_k, seq_len d_k -> ... seq_len d_k').astype(m.dtype)
    b = op.einsum(n, sin_val, '... seq_len d_k, seq_len d_k -> ... seq_len d_k').astype(m.dtype)
    return a + b
