from itertools import repeat
import jax.random as rand

def split_key_nullable(key: rand.KeyArray | None, num: int=2):
    if key is None:
        return tuple(repeat(None, num))
    else:
        return rand.split(key, num)
