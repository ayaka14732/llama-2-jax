import jax.numpy as jnp
import torch
from transformers import LlamaModel

# model = LlamaModel.from_pretrained('../llama-weights/converted/7B')

# with torch.no_grad():
#     params = {}

#     params['embedding'] = jnp.asarray(model.embed_tokens.weight.numpy())


def convert_params():
    pass
