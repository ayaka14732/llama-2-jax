import jax
from jax import Array
import jax.numpy as jnp

# logits: (batch_size, vocab_size)  jnp.float_
# seq: (batch_size, seq_len)  jnp.uint16
# attn_mask: (batch_size, seq_len)  jnp.bool_
# out: (batch_size, seq_len, vocab_size)  jnp.float_
def penalize_presence(logits: Array, seq: Array, attn_mask: Array, presence_penalty: float) -> Array:
    *_, vocab_size = logits.shape
    counts = jax.vmap(lambda a, m: jnp.bincount(a, weights=m.astype(jnp.uint16), length=vocab_size))(seq, attn_mask)  # (batch_size, vocab_size)
    return logits - (counts > 0).astype(jnp.float_) * presence_penalty  # TODO: optimize (?)
