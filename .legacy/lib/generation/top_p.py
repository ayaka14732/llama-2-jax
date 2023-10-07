import jax
from jax import Array
import jax.nn as nn
import jax.numpy as jnp
import jax.random as rand
from operator import getitem
from typing import NamedTuple

from ..llama import ModelConfig, Llama, forward_llama_model
from ._utils import while_loop

class _TopPGenerationState(NamedTuple):
    key: Array
    seq: Array
    attn_mask: Array
    last_positions: Array
    last_tokens: Array
    finished: Array

class TopPGenerationConfig(NamedTuple):
    eos_token_id: int
    max_length: int
    top_p: float

def _loop_body_top_p(params: Llama, state: _TopPGenerationState, model_config: ModelConfig, top_p_config: TopPGenerationConfig) -> _TopPGenerationState:
    key, seq, attn_mask, last_positions, last_tokens, finished = state
    eos_token_id, max_length, top_p = top_p_config

    outputs = forward_llama_model(params.model, seq, attn_mask, key=None, model_config=model_config)
    logits = jax.vmap(getitem)(outputs, last_positions) @ params.lm_head

    batch_size, vocab_size = logits.shape

    indices = jnp.broadcast_to(jnp.arange(vocab_size, dtype=jnp.uint16), (batch_size, vocab_size))
    sorted_logits, sorted_indices = jax.lax.sort_key_val(-logits, indices, is_stable=False)
    sorted_logits = -sorted_logits
    sorted_probs = nn.softmax(sorted_logits)
    cum_probs = jnp.cumsum(sorted_probs, axis=-1)
    threshold_probs = jnp.maximum(cum_probs[:, 0], top_p)  # guarantee that at least one token will not be masked
    masked_sorted_logits = jnp.where(cum_probs >= threshold_probs[:, None], -jnp.inf, sorted_logits)

    current_positions = jnp.where(finished, last_positions, last_positions + 1)

    key, subkey = rand.split(key)
    selected_indices = rand.categorical(subkey, masked_sorted_logits)
    current_tokens = jax.vmap(getitem)(sorted_indices, selected_indices)  # type: ignore[call-overload]
    current_tokens = jnp.where(finished, last_tokens, current_tokens)

    seq = jax.vmap(lambda x, y, z: x.at[y].set(z))(seq, current_positions, current_tokens)
    attn_mask = jax.vmap(lambda x, y: x.at[y].set(True))(attn_mask, current_positions)

    finished |= (current_positions == max_length - 1) | (current_tokens == eos_token_id)

    return _TopPGenerationState(key, seq, attn_mask, current_positions, current_tokens, finished)

def top_p(params: Llama, seq: Array, attn_mask: Array, *, key: Array, model_config: ModelConfig, top_p_config: TopPGenerationConfig) -> Array:
    assert 0.0 < top_p_config.top_p < 1.0

    last_positions = jnp.argmin(attn_mask, axis=-1) - 1
    last_tokens = jax.vmap(getitem)(seq, last_positions)  # type: ignore[call-overload]
    finished = jnp.all(attn_mask, axis=-1)
    loop_body = lambda state: _loop_body_top_p(params, state, model_config, top_p_config)
    state = _TopPGenerationState(key, seq, attn_mask, last_positions, last_tokens, finished)

    state = while_loop((lambda state: ~jnp.all(state.finished)), loop_body, state)
    return state.seq
