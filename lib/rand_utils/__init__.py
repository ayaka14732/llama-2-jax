from itertools import repeat
import jax.random as rand
from typing import Optional

def split_key(key: Optional[rand.KeyArray], num: int=2):
    if key is None:
        return (*repeat(None, num),)
    else:
        return rand.split(key, num)