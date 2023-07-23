from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from functools import partial
import jax
import jax.numpy as jnp

class Whatever:
    def __init__(self, x: str) -> None:
        self.x = x

@partial(jax.jit, static_argnums=(0,))
def f(whatever, a, b):
    if whatever.x == '1':
        return a + b
    else:
        return a - b

@jax.jit
def g(a):
    return a

a = jnp.array([1, 2])
b = jnp.array([3, 0])

assert jnp.array_equal(f(Whatever('1'), a, b), jnp.array([4, 2]))
assert jnp.array_equal(f(Whatever('2'), a, b), jnp.array([-2, 2]))
