from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from flax.serialization import msgpack_restore, msgpack_serialize
from jax import Array
import numpy as np
from typing import NamedTuple

from lib.array_utils import np2jax

class Apple(NamedTuple):
    a: Array
    b: Array

class Banana(NamedTuple):
    a: Apple
    b: Array

a0 = np2jax(np.random.rand(1, 5))
a1 = np2jax(np.random.rand(2, 7))
a2 = np2jax(np.random.rand(4, 8))

params = Banana(a=Apple(a=a0, b=a1), b=a2)
restored_params = msgpack_restore(msgpack_serialize(params))
assert restored_params == params
