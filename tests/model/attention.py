from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
import numpy as np
from transformers.models.llama.modeling_llama import LlamaAttention

from lib.array_utils import np2jax, np2pt, pt2jax
from lib.model import config_7B, rms_norm

batch_size = 2
seq_len = 9
d_model = config_7B.d_model
rms_norm_eps = config_7B.rms_norm_eps

# TODO
