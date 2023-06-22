from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from lib.proc_init_utils import initialise_cpu; initialise_cpu()

import jax.numpy as jnp
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

from lib.array_utils import pt2jax
from lib.model.rotary_embedding import make_weights

batch_size = 2
seq_len = 2048
d_kv = 512

rotary_embedding_pt = LlamaRotaryEmbedding(dim=d_kv, max_position_embeddings=seq_len)
sin_val_pt = rotary_embedding_pt.sin_cached
cos_val_pt = rotary_embedding_pt.cos_cached
sin_val_jax = pt2jax(sin_val_pt)
cos_val_jax = pt2jax(cos_val_pt)

sin_val_hat_jax, cos_val_hat_jax = make_weights(seq_len, d_kv)

assert jnp.allclose(sin_val_jax.squeeze(), sin_val_hat_jax)
assert jnp.allclose(cos_val_jax.squeeze(), cos_val_hat_jax)
