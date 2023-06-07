import jax.numpy as jnp
import torch
from transformers import LlamaModel

from lib.array_utils import pt2jax
from lib.model.llama import Llama

def convert_params():
    model = LlamaModel.from_pretrained('../llama-weights/converted/7B')
    with torch.no_grad():
        return Llama(
            embedding=pt2jax(model.embed_tokens.weight),
            decoder=...,
            norm=...,
        )
