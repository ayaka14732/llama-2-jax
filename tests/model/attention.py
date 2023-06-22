from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from lib.array_utils import pt2jax
from lib.model import attention, config_7B
from lib.param_utils import convert_attention

batch_size = 2  # TODO: test masking
seq_len = 7
d_model = 20
d_k = 10
d_v = 10
n_heads = 2

config_pt = LlamaConfig(hidden_size=d_model, num_attention_heads=n_heads)  # default to LLaMA 7B
config_jax = config_7B._replace(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v)

attention_pt = LlamaAttention(config=config_pt)
params_jax = convert_attention(attention_pt, config=config_jax)

# initialise input sequence
seq_pt = torch.rand(batch_size, seq_len, d_model)
seq_jax = pt2jax(seq_pt)

y_pt = attention_pt(hidden_states=seq_pt, attention_mask=None)[0]
y_jax = pt2jax(y_pt)
y_head_jax = attention(params_jax, seq_jax, seq_jax, None, config=config_jax)
print('y_jax', y_jax[0, 0])
print('y_hea', y_head_jax[0, 0])
assert jnp.allclose(y_jax, y_head_jax)
