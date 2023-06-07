from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from jax import Array
import jax.numpy as jnp

a = jnp.array([2., 5.])
assert isinstance(a, Array)
assert a.shape == (2,)
