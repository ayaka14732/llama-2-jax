# from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_tpu; initialise_tpu('v4-16', n_devices=1, rank=0)

import jax
import jax.numpy as jnp
import jax.random as rand
from jax_smi import initialise_tracking
from operator import getitem
from transformers import LlamaTokenizer

from lib.model import llama_model, config_7B
from lib.param_utils import load_params
from lib.seeding import BEST_INTEGER

initialise_tracking()

top_k = 10
max_length = 256

tokenizer = LlamaTokenizer.from_pretrained('../llama-weights')
params = load_params('7B.pickle')

tokenizer.pad_token = tokenizer.eos_token
inputs = tokenizer(['Simply put, the theory of relativity states that'] * 2, max_length=max_length, padding='max_length', return_tensors='jax')
seq = inputs.input_ids.astype(jnp.uint16)
attn_mask = inputs.attention_mask.astype(jnp.bool_)

key = rand.PRNGKey(BEST_INTEGER)
initial_position = jnp.argmin(attn_mask).item() - 1

def loop_body(i, state):
    key, seq, attn_mask = state

    outputs = llama_model(params.model, seq, attn_mask, config=config_7B)
    logits = outputs[:, i] @ params.lm_head

    values, indices = jax.lax.top_k(logits, k=top_k)
    indices = indices.astype(jnp.uint16)

    key, subkey = rand.split(key)
    selected_indices = rand.categorical(subkey, values)
    generated_tokens = jax.vmap(getitem)(indices, selected_indices)  # type: ignore[call-overload]

    seq = seq.at[:, i+1].set(generated_tokens)
    attn_mask = attn_mask.at[:, i+1].set(True)

    return key, seq, attn_mask

for i in range(initial_position, max_length - 1):
    key, seq, attn_mask = loop_body(i, (key, seq, attn_mask))

decoded_texts = tokenizer.batch_decode(seq, skip_special_tokens=True)
for decoded_text in decoded_texts:
    print(decoded_text, end='\n\n')
