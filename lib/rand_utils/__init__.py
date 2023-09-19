from itertools import repeat
from jax import Array
import jax.random as rand

def split_key_nullable(key: Array | None, num: int=2):
    if key is None:
        return tuple(repeat(None, num))
    else:
        return rand.split(key, num)
