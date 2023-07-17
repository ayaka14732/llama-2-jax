from jax import Array
import jax.numpy as jnp
import optax

def cross_entropy_loss(logits: Array, labels: Array, *, mask: Array) -> Array:
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    return jnp.mean(loss, where=mask)
