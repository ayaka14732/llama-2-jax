from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
import numpy as np
import torch
import torch.nn as tnn
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from lib.array_utils import pt2jax
from lib.model import config_7B, Attention, attention

batch_size = 1  # TODO: change to 2 to test masking
seq_len = 9

config_pt = LlamaConfig()  # default to LLaMA 7B

attention_pt = LlamaAttention(config=config_pt)

params_jax = convert_attention()

config_7B