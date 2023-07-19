from functools import partial
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
from typing import NamedTuple

from .Config import Config
from .decoder import Decoder, check_decoder, decoder
from .embedding import check_embedding, embedding
from .rms_norm import check_rms_norm, rms_norm

class LlamaModel(NamedTuple):
    embedding: Array
    decoder: Decoder
    norm: Array

def check_llama_model(params: LlamaModel, *, config: Config) -> None:
    assert isinstance(params.embedding, Array)
    assert isinstance(params.decoder, list)
    assert isinstance(params.norm, Array)

    check_embedding(params.embedding, config=config)
    check_decoder(params.decoder, config=config)
    check_rms_norm(params.norm, config=config)

@partial(jax.jit, static_argnames=('config'))
def llama_model(params: LlamaModel, seq: Array, attn_mask: Array, *, key: rand.KeyArray, config: Config) -> Array:
    assert isinstance(seq, Array)
    assert isinstance(attn_mask, Array)
    assert seq.dtype == jnp.uint16
    assert attn_mask.dtype == jnp.bool_
    assert seq.shape == attn_mask.shape
    assert config.d_k % 2 == 0
    assert config.n_heads_kv * config.n_rep_kv == config.n_heads_q
    assert key is None or config.dropout_rate is not None

    attn_mask = jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask))[:, None, None]

    seq = embedding(params.embedding, seq)
    seq = decoder(params.decoder, seq, attn_mask, key=key, config=config)
    seq = rms_norm(params.norm, seq, config=config)
    return seq
