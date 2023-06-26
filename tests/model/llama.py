from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
import torch
from transformers import LlamaConfig, LlamaModel

from lib.array_utils import pt2jax
from lib.model import check_llama, config_7B, llama
from lib.param_utils import convert_llama
from lib.seeding import BEST_INTEGER

batch_size = 2
seq_len = 7
d_model = 20
d_k = 10
d_v = 10
d_ff = 13
n_heads = 2
vocab_size = 42

torch.manual_seed(BEST_INTEGER)

vocab_size = 42
config_pt = LlamaConfig(vocab_size=vocab_size, hidden_size=d_model, num_attention_heads=n_heads, intermediate_size=d_ff)
config_jax = config_7B._replace(d_model=d_model, n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, vocab_size=vocab_size)

llama_pt = LlamaModel(config=config_pt)
params_jax = convert_llama(llama_pt, config=config_jax)
check_llama(params_jax, config=config_jax)

# initialise input sequence
seq_pt = (torch.rand(batch_size, seq_len) * config_jax.vocab_size).to(torch.int32)
seq_jax = pt2jax(seq_pt).astype(jnp.uint16)

mask_pt_1d = torch.rand(batch_size, seq_len) > 0.5
mask_jax_1d = pt2jax(mask_pt_1d)

y_pt = llama_pt(input_ids=seq_pt, attention_mask=mask_pt_1d)[0]
y_jax = pt2jax(y_pt)
y_hat_jax = llama(params_jax, seq_jax, mask_jax_1d, config=config_jax)

y_jax = jnp.where(mask_jax_1d[..., None], y_jax, 0.)
y_hat_jax = jnp.where(mask_jax_1d[..., None], y_hat_jax, 0.)

print('y_jax', y_jax.reshape(-1)[:30])
print('y_hat_jax', y_hat_jax.reshape(-1)[:30])
assert jnp.allclose(y_jax, y_hat_jax)
