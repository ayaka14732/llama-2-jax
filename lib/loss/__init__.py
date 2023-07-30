from jax import Array
import jax.numpy as jnp
import optax

def cross_entropy_loss(logits: Array, labels: Array, *, mask: Array) -> Array:
    '''
    Computes the mean softmax cross entropy loss for the provided logits and labels, only considering elements where the mask is True.

    Args:
        logits (Array): The model's predictions. Typically an array of shape (batch_size, seq_len, vocab_size).
        labels (Array): The true labels. Typically an array of shape (batch_size, seq_len).
        mask (Array): Typically a boolean array of shape (batch_size, seq_len). Specifies which elements of the loss array should be included in the mean loss calculation.

    Returns:
        Array: The mean masked softmax cross entropy loss.

    Notes:
        The mask array typically specifies which elements of the sequence are actual tokens as opposed to padding tokens, so the loss is not calculated over padding tokens.
    '''
    loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=labels)
    return jnp.mean(loss, where=mask)
