from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaAttention

from lib.array_utils import pt2jax
from lib.LLM import model_config_llama1_7B
from lib.LLM.attention import forward_attention
from lib.LLM_params.convert_params import convert_attention
from lib.seeding import BEST_INTEGER

batch_size = 2
seq_len = 7
d_model = 20
d_k = 10
d_v = 10
n_heads_kv = 2

torch.manual_seed(BEST_INTEGER)

config_pt = LlamaConfig(hidden_size=d_model, num_attention_heads=n_heads_kv)
config_jax = model_config_llama1_7B._replace(d_model=d_model, n_heads_kv=n_heads_kv, d_k=d_k, d_v=d_v, dropout_rate=None)

attention_pt = LlamaAttention(config=config_pt)
params_jax = convert_attention(attention_pt, model_config=config_jax)

# initialise input sequence
seq_pt = torch.rand(batch_size, seq_len, d_model)
seq_jax = pt2jax(seq_pt)

mask_pt_1d = torch.ones(batch_size, seq_len, dtype=torch.bool)  # torch.rand(batch_size, seq_len) > 0.1
mask_pt = torch.tril(torch.einsum('bi,bj->bij', mask_pt_1d, mask_pt_1d))[:, None]
mask_jax_1d = pt2jax(mask_pt_1d)
mask_jax = jnp.tril(jnp.einsum('bi,bj->bij', mask_jax_1d, mask_jax_1d))[:, None, None]

# In the Hugging Face implementation, the attention mask is added to the attention
# matrix, not multiplied.
# See https://github.com/huggingface/transformers/issues/1935
mask_pt = torch.where(mask_pt, 0, -10000.)

y_pt = attention_pt(hidden_states=seq_pt, attention_mask=mask_pt)[0]
y_jax = pt2jax(y_pt)
y_hat_jax, _ = forward_attention(params_jax, seq_jax, seq_jax, mask_jax, model_config=config_jax)

y_jax = jnp.where(mask_jax_1d[..., None], y_jax, 0.)
y_hat_jax = jnp.where(mask_jax_1d[..., None], y_hat_jax, 0.)

print('y_jax', y_jax.reshape(-1)[:30])
print('y_hat_jax', y_hat_jax.reshape(-1)[:30])
assert jnp.allclose(y_jax, y_hat_jax)
