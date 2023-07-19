import jax
from jax import Array
import jax.numpy as jnp

def penalize_presence(logits: Array, seq: Array, attn_mask: Array, presence_penalty: float) -> Array:
    # TODO: need type checking?
    batch_size, vocab_size = logits.shape
    _, seq_len = seq.shape
    assert seq.shape == (batch_size, seq_len)
    assert seq.dtype == jnp.uint16
    assert attn_mask.shape == (batch_size, seq_len)
    assert attn_mask.dtype == jnp.bool_

    exists = jax.vmap(lambda row, row_mask: jnp.bincount(row, weights=row_mask.astype(jnp.uint16), length=vocab_size).astype(jnp.bool_))(seq, attn_mask)  # (batch_size, vocab_size)
    return logits - exists * presence_penalty
