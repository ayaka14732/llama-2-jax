from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
import numpy as np

from lib.param_utils import check_params_equal

assert not check_params_equal(np.zeros((2, 3)), jnp.zeros((2, 3)))
