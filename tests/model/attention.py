from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from lib.array_utils import pt2jax
from lib.model import config_7B, attention
from lib.param_utils import convert_attention

batch_size = 2  # TODO: change to 2 to test masking
seq_len = 7

config_pt = LlamaConfig()  # default to LLaMA 7B

attention_pt = LlamaAttention(config=config_pt)

params_jax = convert_attention(attention_pt, config=config_7B)

# initialise input sequence
seq_pt = torch.rand(batch_size, seq_len, config_7B.d_model)
seq_jax = pt2jax(seq_pt)

y_pt = attention_pt(hidden_states=seq_pt, attention_mask=None)[0]
y_jax = pt2jax(y_pt)
y_head_jax = attention(params_jax, seq_jax, seq_jax, None, config=config_7B)
print('y_jax', y_jax)
print('y_head_jax', y_head_jax)
assert jnp.allclose(y_jax, y_head_jax)
