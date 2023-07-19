import jax
from jax import Array
import jax.numpy as jnp

def penalize_frequency(logits: Array, seq: Array, attn_mask: Array, frequency_penalty: float) -> Array:
    *_, vocab_size = logits.shape
    counts = jax.vmap(lambda row, row_mask: jnp.bincount(row, weights=row_mask.astype(jnp.uint16), length=vocab_size))(seq, attn_mask)  # (batch_size, vocab_size)
    return logits - counts * frequency_penalty
