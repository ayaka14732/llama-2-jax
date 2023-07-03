# from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_tpu; initialise_tpu('v4-16', n_devices=1, rank=0)

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as rand
from jax_smi import initialise_tracking
from operator import getitem
from transformers import LlamaTokenizer
from typing import NamedTuple

from lib.model import config_7B, llama_model
from lib.param_utils import load_params
from lib.seeding import BEST_INTEGER

initialise_tracking()

top_k = 10
max_length = 128

tokenizer = LlamaTokenizer.from_pretrained('../llama-weights')
params = load_params('7B.pickle')

tokenizer.pad_token = tokenizer.eos_token  # TODO: verify this
sentences = [
    'I believe the meaning of life is',
    'Simply put, the theory of relativity states that',
    'Thus, leveraging the potential of quantum computing, we can optimize complex algorithms, paving the way for breakthroughs in fields ranging from cryptography to molecular modeling',
]

class State(NamedTuple):
    key: rand.KeyArray
    seq: Array
    attn_mask: Array
    last_positions: Array
    last_tokens: Array
    finished: Array

# TODO: https://jax.readthedocs.io/en/latest/_autosummary/jax.lax.while_loop.html
def while_loop(cond_fun, body_fun, init_val):
    val = init_val
    while cond_fun(val):
        val = body_fun(val)
    return val

def _loop_body_top_k(state: State) -> State:
    key, seq, attn_mask, last_positions, last_tokens, finished = state

    outputs = llama_model(params.model, seq, attn_mask, config=config_7B)
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

    finished |= (current_positions == max_length - 1) | (current_tokens == tokenizer.eos_token_id)

    return State(key=key, seq=seq, attn_mask=attn_mask, last_positions=current_positions, last_tokens=current_tokens, finished=finished)

def generate_top_k(key: rand.KeyArray, seq: Array, attn_mask: Array) -> Array:
    last_positions = jnp.argmin(attn_mask, axis=-1) - 1
    last_tokens = jax.vmap(getitem)(seq, last_positions)  # type: ignore[call-overload]
    finished = jnp.all(attn_mask, axis=-1)
    state = State(key=key, seq=seq, attn_mask=attn_mask, last_positions=last_positions, last_tokens=last_tokens, finished=finished)
    state = while_loop((lambda state: ~jnp.all(state.finished)), _loop_body_top_k, state)
    return state.seq

def main() -> None:
    key = rand.PRNGKey(BEST_INTEGER)
    inputs = tokenizer(sentences, max_length=max_length, padding='max_length', return_tensors='jax')
    seq = inputs.input_ids.astype(jnp.uint16)
    attn_mask = inputs.attention_mask.astype(jnp.bool_)

    key, subkey = rand.split(key)
    generated_seq = generate_top_k(subkey, seq, attn_mask)
    decoded_texts = tokenizer.batch_decode(generated_seq, skip_special_tokens=True)

    for decoded_text in decoded_texts:
        print(decoded_text, end='\n\n')

if __name__ == '__main__':
    main()
