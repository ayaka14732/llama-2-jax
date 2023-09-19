from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
from typing import Any, NamedTuple

from .ModelConfig import ModelConfig
from .decoder import Decoder, check_decoder, forward_decoder, init_decoder
from .embedding import check_embedding, forward_embedding, init_embedding
from .rms_norm import check_rms_norm, forward_rms_norm, init_rms_norm

class LlamaModel(NamedTuple):
    embedding: Any  # Array
    decoder: Decoder
    norm: Any  # Array

def check_llama_model(params: LlamaModel, *, model_config: ModelConfig) -> None:
    assert isinstance(params.embedding, Array)
    assert isinstance(params.decoder, Decoder)
    assert isinstance(params.norm, Array)

    check_embedding(params.embedding, model_config=model_config)
    check_decoder(params.decoder, model_config=model_config)
    check_rms_norm(params.norm, model_config=model_config)

def init_llama_model(*, key: Array, model_config: ModelConfig) -> LlamaModel:
    key0, key1 = rand.split(key)
    embedding = init_embedding(key=key0, model_config=model_config)
    decoder = init_decoder(key=key1, model_config=model_config)
    norm = init_rms_norm(model_config=model_config)
    return LlamaModel(embedding, decoder, norm)

@partial(jax.jit, static_argnames=('model_config'))
def forward_llama_model(params: LlamaModel, seq: Array, attn_mask: Array, *, key: Array | None, model_config: ModelConfig) -> Array:
    assert isinstance(seq, Array)
    assert isinstance(attn_mask, Array)
    assert seq.dtype == jnp.uint16
    assert attn_mask.dtype == jnp.bool_
    assert seq.shape == attn_mask.shape
    assert model_config.d_k % 2 == 0
    assert key is None or model_config.dropout_rate is not None

    attn_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask))[:, None, None]

    seq = forward_embedding(params.embedding, seq)
    seq = forward_decoder(params.decoder, seq, attn_mask, key=key, model_config=model_config)
    seq = forward_rms_norm(params.norm, seq, model_config=model_config)
    return seq
