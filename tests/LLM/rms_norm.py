from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
import torch
import torch.nn as tnn
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from lib.array_utils import pt2jax
from lib.LLM import model_config_llama1_7B
from lib.LLM.rms_norm import forward_rms_norm

batch_size = 2
seq_len = 9

rms_norm_pt = LlamaRMSNorm(hidden_size=model_config_llama1_7B.d_model, eps=model_config_llama1_7B.rms_norm_eps)
rms_norm_pt.weight = tnn.Parameter(torch.randn_like(rms_norm_pt.weight))

params_pt = rms_norm_pt.weight
params_jax = pt2jax(params_pt)

x_pt = torch.rand(batch_size, seq_len, model_config_llama1_7B.d_model)
x_jax = pt2jax(x_pt)

y_pt = rms_norm_pt(x_pt)
y_jax = pt2jax(y_pt)
y_hat_jax = forward_rms_norm(params_jax, x_jax, model_config=model_config_llama1_7B)
assert jnp.allclose(y_jax, y_hat_jax)
