from jax import Array
import jax.numpy as jnp
import numpy as np
from typing import Any

def check_params_equal(t1: Any, t2: Any) -> bool:
    '''
    Recursively checks the equality of two objects. 

    If both objects are NumPy arrays, `np.array_equal()` is used for comparison.
    If both objects are JAX arrays, `jnp.array_equal()` is used for comparison.
    If both objects are namedtuples, the function is called recursively on each corresponding field.
    Otherwise, the standard equality operator `==` is used.
    '''
    if t1.__class__ != t2.__class__:
        return False
    if isinstance(t1, np.ndarray):
        return np.array_equal(t1, t2)
    if isinstance(t1, Array):
        return bool(jnp.array_equal(t1, t2))
    if isinstance(t1, tuple) and hasattr(t1, '_fields'):  # check if t1 is namedtuple
        return all(check_params_equal(t1[i], t2[i]) for i in range(len(t1)))
    return t1 == t2
