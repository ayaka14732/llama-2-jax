from lib.proc_init_utils import initialise_tpu; initialise_tpu('v3-32', n_devices=8)

import einops as op
import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
from transformers import LlamaTokenizer

from lib.llama import Llama, RotaryValues, forward_llama_model, make_rotary_values, model_config_llama2_7B, shift_left_kv_cache, shift_left_rotary_values
from lib.param_utils import load_params
from lib.multihost_utils import shard_model_params
from lib.seeding import HASHED_BUDDHA

max_len = 128
top_k = 4

def load_params_from_disk() -> Llama:
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        params = load_params('llama2-7B.pickle')
        params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    params = shard_model_params(params)
    return params

def top_k_sampling_from_logits(logits: Array, *, key: Array) -> Array:
    batch_size, *_ = logits.shape
    keys = rand.split(key, num=batch_size)

    def inner(logits: Array, *, keys: Array) -> Array:
        values, indices = jax.lax.top_k(logits, k=top_k)
        indices = indices.astype(jnp.uint16)
        selected_index = rand.categorical(key, values)
        selected_token_id = indices[selected_index]
        return selected_token_id

    selected_token_ids = jax.vmap(inner)(logits, keys=keys)
    return selected_token_ids

def generate(sentences: list[str], tokenizer: LlamaTokenizer, params: Llama, *, key: Array) -> list[str]:
    batch_size = len(sentences)

    tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(sentences, padding='max_length', truncation=True, max_length=max_len, return_tensors='jax')
    seq = inputs.input_ids.astype(jnp.uint16)
    attn_mask = inputs.attention_mask.astype(jnp.bool_)
    qk_mask = op.rearrange(jnp.tril(jnp.einsum('bi,bj->bij', attn_mask, attn_mask)), 'B L1 L2 -> B 1 1 L1 L2')
    assert not attn_mask.all(axis=-1).any()  # every sample has room to generate

    # initial forward operation
    initial_seq_len = seq.shape[1]
    leftpad_len = (~attn_mask).argmin(-1).astype(jnp.uint16)
    rotary_values = make_rotary_values(leftpad_len, batch_size, max_len, model_config=model_config_llama2_7B)
    # print(rotary_values.sin_val.shape)

    outputs, kv_cache = forward_llama_model(params.model, seq, qk_mask, rotary_values=rotary_values, model_config=model_config_llama2_7B._replace(return_kv_cache=True))
    logits = outputs[:, -1] @ params.lm_head

    key, subkey = rand.split(key)
    selected_token_ids = top_k_sampling_from_logits(logits, key=subkey)
    # print(tokenizer.batch_decode(selected_token_ids, skip_special_tokens=True))

    # consequent forward operations
    max_n_iters = leftpad_len.min().item()

    for _ in range(max_n_iters):
        seq = jnp.roll(seq, -1, axis=-1).at[:, -1].set(selected_token_ids)  # TODO: optimise
        attn_mask = jnp.roll(attn_mask, -1, axis=-1).at[:, -1].set(True)
        rotary_values = shift_left_rotary_values(rotary_values)

        seq_ = selected_token_ids[:, None]
        qk_mask = op.rearrange(attn_mask, 'B L -> B 1 1 1 L')

        sin_val, cos_val = rotary_values
        rotary_values_ = RotaryValues(sin_val[:, -1:], cos_val[:, -1:])
        kv_cache = shift_left_kv_cache(kv_cache)

        outputs, kv_cache = forward_llama_model(params.model, seq_, qk_mask, rotary_values=rotary_values_, kv_cache=kv_cache, model_config=model_config_llama2_7B._replace(return_kv_cache=True))
        logits = outputs[:, -1] @ params.lm_head

        key, subkey = rand.split(key)
        selected_token_ids = top_k_sampling_from_logits(logits, key=subkey)
        # print(tokenizer.batch_decode(selected_token_ids, skip_special_tokens=True))

    return tokenizer.batch_decode(seq, skip_special_tokens=True)

def main():
    params = load_params_from_disk()
    print('Successfully loaded model parameters!')
    key = rand.key(HASHED_BUDDHA, impl='rbg')
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', padding_side='left')
    sentences = [
        'Four score and seven years ago our fathers brought forth',
        'I go to school by',
    ]
    generated_sentences = generate(sentences, tokenizer, params, key=key)
    print(generated_sentences)

if __name__ == '__main__':
    main()
