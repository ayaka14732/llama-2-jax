from jax import Array

def bias(logits: Array, bias: float) -> Array:
    return logits + bias
