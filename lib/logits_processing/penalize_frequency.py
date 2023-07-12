import jax
from jax import Array
import jax.numpy as jnp

def penalize_frequency(logits: Array, seq: Array, attn_mask: Array, frequency_penalty: float) -> Array:
    *_, vocab_size = logits.shape
    counts = jax.vmap(lambda a, m: jnp.bincount(a, weights=m.astype(jnp.uint16), length=vocab_size))(seq, attn_mask)  # (batch_size, vocab_size)
    return logits - counts.astype(jnp.float_) * frequency_penalty
