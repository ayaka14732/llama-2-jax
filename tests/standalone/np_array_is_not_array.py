from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

from jax import Array
import numpy as np

assert not isinstance(np.zeros((2, 3)), Array)
