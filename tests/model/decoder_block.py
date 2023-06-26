from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
import torch
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

from lib.array_utils import pt2jax
from lib.model import config_7B, decoder_block
from lib.param_utils import convert_decoder_block
from lib.seeding import BEST_INTEGER

batch_size = 2
seq_len = 7
d_model = 20
d_k = 10
d_v = 10
d_ff = 13
n_heads = 2

torch.manual_seed(BEST_INTEGER)

config_pt = LlamaConfig(hidden_size=d_model, num_attention_heads=n_heads, intermediate_size=d_ff)
config_jax = config_7B._replace(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff)

decoder_block_pt = LlamaDecoderLayer(config=config_pt)
params_jax = convert_decoder_block(decoder_block_pt, config=config_jax)

# initialise input sequence
seq_pt = torch.rand(batch_size, seq_len, d_model)
seq_jax = pt2jax(seq_pt)

mask_pt_1d = torch.rand(batch_size, seq_len) > 0.5
mask_pt = torch.tril(torch.einsum('di,dj->dij', mask_pt_1d, mask_pt_1d))[:, None]
mask_jax_1d = pt2jax(mask_pt_1d)
mask_jax = pt2jax(mask_pt)

# In the Hugging Face implementation, the attention mask is added to the attention
# matrix, not multiplied.
# See https://github.com/huggingface/transformers/issues/1935
mask_pt = torch.where(mask_pt, 0, -10000.)

y_pt = decoder_block_pt(hidden_states=seq_pt, attention_mask=mask_pt)[0]
y_jax = pt2jax(y_pt)
y_hat_jax = decoder_block(params_jax, seq_jax, mask_jax, config=config_jax)

y_jax = jnp.where(mask_jax_1d[..., None], y_jax, 0.)
y_hat_jax = jnp.where(mask_jax_1d[..., None], y_hat_jax, 0.)

print('y_jax', y_jax.reshape(-1)[:30])
print('y_hat_jax', y_hat_jax.reshape(-1)[:30])
assert jnp.allclose(y_jax, y_hat_jax)
