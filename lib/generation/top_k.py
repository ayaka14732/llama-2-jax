import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
from operator import getitem
from typing import NamedTuple

from ..llama import ModelConfig, Llama, forward_llama_model
from ._utils import while_loop

class _TopKGenerationState(NamedTuple):
    key: Array
    seq: Array
    attn_mask: Array
    last_positions: Array
    last_tokens: Array
    finished: Array

class TopKGenerationConfig(NamedTuple):
    eos_token_id: int
    max_length: int
    top_k: int

def _loop_body_top_k(params: Llama, state: _TopKGenerationState, model_config: ModelConfig, top_k_config: TopKGenerationConfig) -> _TopKGenerationState:
    key, seq, attn_mask, last_positions, last_tokens, finished = state
    eos_token_id, max_length, top_k = top_k_config

    outputs = forward_llama_model(params.model, seq, attn_mask, key=None, model_config=model_config)
    logits = jax.vmap(getitem)(outputs, last_positions) @ params.lm_head

    values, indices = jax.lax.top_k(logits, k=top_k)
    indices = indices.astype(jnp.uint16)

    current_positions = jnp.where(finished, last_positions, last_positions + 1)

    key, subkey = rand.split(key)
    selected_indices = rand.categorical(subkey, values)
    current_tokens = jax.vmap(getitem)(indices, selected_indices)  # type: ignore[call-overload]
    current_tokens = jnp.where(finished, last_tokens, current_tokens)

    seq = jax.vmap(lambda x, y, z: x.at[y].set(z))(seq, current_positions, current_tokens)
    attn_mask = jax.vmap(lambda x, y: x.at[y].set(True))(attn_mask, current_positions)

    finished |= (current_positions == max_length - 1) | (current_tokens == eos_token_id)

    return _TopKGenerationState(key, seq, attn_mask, current_positions, current_tokens, finished)

def top_k(params: Llama, seq: Array, attn_mask: Array, *, key: Array, model_config: ModelConfig, top_k_config: TopKGenerationConfig) -> Array:
    assert top_k_config.top_k > 0

    last_positions = jnp.argmin(attn_mask, axis=-1) - 1
    last_tokens = jax.vmap(getitem)(seq, last_positions)  # type: ignore[call-overload]
    finished = jnp.all(attn_mask, axis=-1)
    loop_body = lambda state: _loop_body_top_k(params, state, model_config, top_k_config)
    state = _TopKGenerationState(key, seq, attn_mask, last_positions, last_tokens, finished)

    state = while_loop((lambda state: ~jnp.all(state.finished)), loop_body, state)
    return state.seq
