import einops as op
import jax
import jax.numpy as jnp

# TODO: Mostly taken from https://github.com/kingoflolz/mesh-transformer-jax/blob/master/mesh_transformer/layers.py
# and https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L92
def make_weights(seq_len, d_kv):
    inv_freq = 1. / (10000 ** (jnp.arange(0, d_kv, 2) / d_kv))
    sinusoid_inp = op.einsum(jnp.arange(seq_len), inv_freq, 'seq_len, j -> seq_len j')
    sin_val = jnp.sin(sinusoid_inp)
    cos_val = jnp.cos(sinusoid_inp)
    sin_val = op.repeat(sin_val, 'seq_len d_kv -> seq_len (i d_kv)', i=2)
    cos_val = op.repeat(cos_val, 'seq_len d_kv -> seq_len (i d_kv)', i=2)
    return sin_val, cos_val

def rotate_every_two(x):
    x = op.rearrange(x, '... (x i) -> ... x i', i=2)
    x = x[..., ::-1] * jnp.array([-1, 1])
    x = op.rearrange(x, '... x i -> ... (x i)')
    return x

def rotary_embedding(x):
    # assert config.d_k % 2 == 0
    # assert config.d_v % 2 == 0

    *_, seq_len, d_kv = x.shape  # `d_kv`: either `d_k` or `d_v` depending on the context

    with jax.ensure_compile_time_eval():
        sin_val, cos_val = make_weights(seq_len, d_kv)

    pattern = 'batch_size n_heads seq_len d_kv, seq_len d_kv -> batch_size n_heads seq_len d_kv'
    a = op.einsum(x, cos_val, pattern)
    b = op.einsum(rotate_every_two(x), sin_val, pattern)
    return a + b
