from functools import partial
from typing import NamedTuple

import einops as op
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
from transformers import LlamaTokenizer
from typing import Callable

from lib.LLM import KVCache, Llama, RotaryValues, forward_llama_model, get_rotary_values_at_position, make_rotary_values, model_config_llama2_7B, shift_left_kv_cache

@partial(jax.jit, static_argnames=('logits_processor',))
def _generate_first(params: Llama, seq: Array, attn_mask: Array, logits_processor: Callable, *, rotary_values: RotaryValues, key: Array) -> tuple[Array, Array, Array, KVCache]:
    qk_mask = op.rearrange(jnp.tril(op.einsum(attn_mask, attn_mask, 'B L1, B L2 -> B L1 L2')), 'B L1 L2 -> B 1 1 L1 L2')  # causal QK mask
    outputs, kv_cache = forward_llama_model(params.model, seq, qk_mask, rotary_values=rotary_values, model_config=model_config_llama2_7B._replace(return_kv_cache=True))

    logits = outputs[:, -1] @ params.lm_head
    selected_token_ids = logits_processor(logits, seq=seq, attn_mask=attn_mask, key=key)

    seq = jnp.roll(seq, -1, axis=-1).at[:, -1].set(selected_token_ids)
    attn_mask = jnp.roll(attn_mask, -1, axis=-1).at[:, -1].set(True)
    kv_cache = shift_left_kv_cache(kv_cache)

    return seq, attn_mask, selected_token_ids, kv_cache

class GenerationState(NamedTuple):
    seq: Array
    attn_mask: Array
    selected_token_ids: Array
    max_n_iters: Array
    rotary_values: RotaryValues
    rotary_values_position: Array
    kv_cache: KVCache
    key: Array

@partial(jax.jit, static_argnames=('logits_processor',))
def _generate_rest(params: Llama, seq: Array, attn_mask: Array, selected_token_ids: Array, max_n_iters: Array, logits_processor: Callable, *, rotary_values: RotaryValues, kv_cache: KVCache, key: Array) -> Array:
    def cond_fun(state: GenerationState) -> Array:
        return state.max_n_iters.astype(jnp.bool_)

    def body_fun(state: GenerationState) -> GenerationState:
        seq, attn_mask, selected_token_ids, max_n_iters, rotary_values, rotary_values_position, kv_cache, key = state

        seq_ = op.rearrange(selected_token_ids, 'B -> B 1')
        qk_mask = op.rearrange(attn_mask, 'B L -> B 1 1 1 L')
        rotary_values_ = get_rotary_values_at_position(rotary_values, rotary_values_position)
        outputs, kv_cache = forward_llama_model(params.model, seq_, qk_mask, rotary_values=rotary_values_, kv_cache=kv_cache, model_config=model_config_llama2_7B._replace(return_kv_cache=True))

        logits = outputs[:, -1] @ params.lm_head
        key, subkey = rand.split(key)
        selected_token_ids = logits_processor(logits, seq=seq, attn_mask=attn_mask, key=subkey)

        seq = jnp.roll(seq, -1, axis=-1).at[:, -1].set(selected_token_ids)
        attn_mask = jnp.roll(attn_mask, -1, axis=-1).at[:, -1].set(True)
        kv_cache = shift_left_kv_cache(kv_cache)

        rotary_values_position += 1
        max_n_iters -= 1
        # TODO: early stopping

        return GenerationState(seq, attn_mask, selected_token_ids, max_n_iters, rotary_values, rotary_values_position, kv_cache, key)

    rotary_values_position = jnp.array(0, jnp.uint16)
    initial_state = GenerationState(seq, attn_mask, selected_token_ids, max_n_iters, rotary_values, rotary_values_position, kv_cache, key)
    final_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    return final_state.seq

def generate(sentences: list[str], tokenizer: LlamaTokenizer, params: Llama, logits_processor: Callable, *, max_len: int, key: Array) -> list[str]:
    batch_size = len(sentences)

    inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=max_len, return_tensors='jax')
    seq = inputs.input_ids.astype(jnp.uint16)
    attn_mask = inputs.attention_mask.astype(jnp.bool_)
    assert not attn_mask.all(axis=-1).any(), 'No room for generation since the length of a sentence is greater than `max_length`.'

    leftpad_len = attn_mask.argmax(axis=-1).astype(jnp.uint16)
    rotary_values = make_rotary_values(leftpad_len, batch_size, max_len, model_config=model_config_llama2_7B)

    key, subkey = rand.split(key)
    seq, attn_mask, selected_token_ids, kv_cache = _generate_first(params, seq, attn_mask, logits_processor, rotary_values=rotary_values, key=subkey)

    max_n_iters = leftpad_len.min()
    key, subkey = rand.split(key)
    seq = _generate_rest(params, seq, attn_mask, selected_token_ids, max_n_iters, logits_processor, rotary_values=rotary_values, kv_cache=kv_cache, key=subkey)

    return tokenizer.batch_decode(seq, skip_special_tokens=True)
