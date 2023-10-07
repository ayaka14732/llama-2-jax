from lib.proc_init_utils import initialise_tpu; initialise_tpu('v3-32', n_devices=8)

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
from transformers import LlamaTokenizer

from lib.llama import forward_llama_model, init_kv_cache, model_config_llama2_7B
from lib.param_utils import load_params
from lib.multihost_utils import shard_model_params
from lib.seeding import HASHED_BUDDHA

max_len = 256
top_k = 1

def load_params_from_disk():
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        params = load_params('llama2-7B.pickle')
        params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    params = shard_model_params(params)
    return params

def top_k_smapling_from_logits(logits: Array, *, key: Array) -> Array:
    values, indices = jax.lax.top_k(logits, k=top_k)
    indices = indices.astype(jnp.uint16)
    selected_indices = rand.categorical(key, values)
    selected_token_ids = indices[0, selected_indices]
    return selected_token_ids

params = load_params_from_disk()

key = rand.key(HASHED_BUDDHA, impl='rbg')

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
sentences = ['Four score and seven years ago our fathers brought forth']  # multiple sentences are currently not supported

batch_size = len(sentences)

tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(sentences, padding='max_length', max_length=max_len, return_tensors='jax')
seq = inputs.input_ids.astype(jnp.uint16)
attn_mask = inputs.attention_mask.astype(jnp.bool_)

kv_cache = init_kv_cache(batch_size, max_len, model_config=model_config_llama2_7B)
cache_position = jnp.array(0, dtype=jnp.uint16)
kv_cache = jax.tree_map(lambda x: x.astype(jnp.bfloat16), kv_cache)

# initial forward
outputs, kv_cache = forward_llama_model(params.model, seq, attn_mask, cache_position=cache_position, kv_cache=kv_cache, key=None, model_config=model_config_llama2_7B)
cache_position = attn_mask.argmin().astype(jnp.uint16)
logits = outputs[:, cache_position - 1] @ params.lm_head

key, subkey = rand.split(key)
selected_token_ids = top_k_smapling_from_logits(logits, key=subkey)

while cache_position < max_len:
    seq = seq.at[:, cache_position].set(selected_token_ids)
    attn_mask = attn_mask.at[:, cache_position].set(True)

    outputs, kv_cache = forward_llama_model(params.model, seq, attn_mask, cache_position=cache_position, kv_cache=kv_cache, key=None, model_config=model_config_llama2_7B)
    cache_position += 1
    logits = outputs[:, cache_position - 1] @ params.lm_head

    key, subkey = rand.split(key)
    selected_token_ids = top_k_smapling_from_logits(logits, key=subkey)

print(tokenizer.batch_decode(seq, skip_special_tokens=True))
