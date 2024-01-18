from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
import torch
import torch.nn as tnn

from lib.array_utils import pt2jax
from lib.LLM import model_config_llama1_7B
from lib.LLM.embedding import forward_embedding

vocab_size = 12
d_model = 2

embedding_pt = tnn.Embedding(model_config_llama1_7B.vocab_size, model_config_llama1_7B.d_model, model_config_llama1_7B.token_id_pad)
embedding_pt.weight = tnn.Parameter(torch.randn_like(embedding_pt.weight))

params_pt = embedding_pt.weight
params_jax = pt2jax(params_pt)

x_pt = torch.tensor([[12, 36, 5145, 21, 0, 556, 25454, 54, 563, 8]], dtype=torch.int)
x_jax = pt2jax(x_pt).astype(jnp.uint16)

y_pt = embedding_pt(x_pt)
y_jax = pt2jax(y_pt)
y_hat_jax = forward_embedding(params_jax, x_jax)
assert jnp.allclose(y_jax, y_hat_jax)
