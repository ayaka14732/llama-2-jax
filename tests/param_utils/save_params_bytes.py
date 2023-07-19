from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from jax import Array
import numpy as np
from typing import NamedTuple

from lib.array_utils import np2jax
from lib.param_utils import check_params_equal, load_params_bytes, save_params_bytes

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
restored_params = load_params_bytes(save_params_bytes(params))
assert check_params_equal(params, restored_params)
