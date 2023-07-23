from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_tpu; initialise_tpu(accelerator_type='v4-16', n_devices=2, rank=0)

import jax
import jax.numpy as jnp
import jax.random as rand

assert jax.process_count() == 1
assert jax.local_device_count() == 2

a = jnp.array([1., 2.])
assert repr(a.device()).startswith('TpuDevice')

key = rand.PRNGKey(42)
assert repr(key).startswith('PRNGKeyArray[rbg] {')
assert repr(key.device()).startswith('TpuDevice')
