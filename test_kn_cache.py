from lib.proc_init_utils import initialise_tpu; initialise_tpu('v3-32', n_devices=8)

import jax
import jax.numpy as jnp
from transformers import LlamaTokenizer

from lib.llama import forward_llama, init_kv_cache, model_config_llama2_7B
from lib.param_utils import load_params
from lib.multihost_utils import shard_model_params

cpu_device = jax.devices('cpu')[0]
with jax.default_device(cpu_device):
    params = load_params('llama2-7B.pickle')
    params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
params = shard_model_params(params)

max_len = 128

tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
sentences = ['The meaning of life is to']

batch_size = len(sentences)

tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(sentences, padding='max_length', max_length=max_len, return_tensors='jax')
seq = inputs.input_ids.astype(jnp.uint16)
attn_mask = inputs.attention_mask.astype(jnp.bool_)

cache_position, kv_cache = init_kv_cache(batch_size, max_len, model_config=model_config_llama2_7B)
kv_cache = jax.tree_map(lambda x: x.astype(jnp.bfloat16), kv_cache)

# first-time forward
logits, kv_cache = forward_llama(params, seq, attn_mask, cache_position=cache_position, kv_cache=kv_cache, key=None, model_config=model_config_llama2_7B)
cache_position = attn_mask.argmin().astype(jnp.uint16)
selected_token_ids = logits[:, cache_position - 1].argmax(-1)
print(tokenizer.convert_ids_to_tokens(selected_token_ids))

seq = seq.at[:, cache_position].set(selected_token_ids)
attn_mask = attn_mask.at[:, cache_position].set(True)

for i in range(30):
    logits, kv_cache = forward_llama(params, seq, attn_mask, cache_position=cache_position, kv_cache=kv_cache, key=None, model_config=model_config_llama2_7B)
    cache_position += 1
    selected_token_ids = logits[:, cache_position - 1].argmax(-1)
    print(tokenizer.convert_ids_to_tokens(selected_token_ids))

    seq = seq.at[:, cache_position].set(selected_token_ids)
    attn_mask = attn_mask.at[:, cache_position].set(True)

print(tokenizer.batch_decode(seq, skip_special_tokens=True))
