from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp

a = jnp.array([[1, 2], [3, 4]])
assert isinstance(a.shape, tuple)
