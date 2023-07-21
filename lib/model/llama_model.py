from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
from typing import NamedTuple

from .ModelConfig import ModelConfig
from .decoder import Decoder, check_decoder, decoder, shard_decoder
from .embedding import check_embedding, embedding, shard_embedding
from .rms_norm import check_rms_norm, rms_norm, shard_rms_norm

class LlamaModel(NamedTuple):
    embedding: Array
    decoder: Decoder
    norm: Array

def check_llama_model(params: LlamaModel, *, model_config: ModelConfig) -> None:
    assert isinstance(params.embedding, Array)
    assert isinstance(params.decoder, Decoder)
    assert isinstance(params.norm, Array)

    check_embedding(params.embedding, model_config=model_config)
    check_decoder(params.decoder, model_config=model_config)
    check_rms_norm(params.norm, model_config=model_config)

def shard_llama_model(params: LlamaModel) -> LlamaModel:
    embedding = shard_embedding(params.embedding)
    decoder = shard_decoder(params.decoder)
    norm = shard_rms_norm(params.norm)
    return LlamaModel(embedding, decoder, norm)

@partial(jax.jit, static_argnames=('model_config'))
def llama_model(params: LlamaModel, seq: Array, attn_mask: Array, *, key: rand.KeyArray, model_config: ModelConfig) -> Array:
    assert isinstance(seq, Array)
    assert isinstance(attn_mask, Array)
    assert seq.dtype == jnp.uint16
    assert attn_mask.dtype == jnp.bool_
    assert seq.shape == attn_mask.shape
    assert model_config.d_k % 2 == 0
    assert model_config.n_heads_kv * model_config.n_rep_kv == model_config.n_heads_q
    assert key is None or model_config.dropout_rate is not None

    attn_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask))[:, None, None]

    seq = embedding(params.embedding, seq)
    seq = decoder(params.decoder, seq, attn_mask, key=key, model_config=model_config)
    seq = rms_norm(params.norm, seq, model_config=model_config)
    return seq
