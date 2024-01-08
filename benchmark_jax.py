from lib.proc_init_utils import initialise_tpu; initialise_tpu('v3-32', n_devices=8)

import jax
import jax.numpy as jnp
import jax.random as rand
import timeit
from transformers import LlamaTokenizer

from lib.generation import generate
from lib.llama import Llama
from lib.logits_processing import PresencePenaltyProcessor, TopKSampler, make_logits_processor
from lib.param_utils import load_params
from lib.multihost_utils import shard_model_params
from lib.seeding import BEST_INTEGER

def load_params_from_disk() -> Llama:
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        params = load_params('llama2-7B.pickle')
        params = jax.tree_map(lambda x: x.astype(jnp.bfloat16), params)
    params = shard_model_params(params)
    return params

def generate_jax(params, tokenizer, logits_processor, max_len, key):
    sentences = ['Four score and seven years ago our fathers'] * 30 + ['Marta is a Syriac student. Aday asks her: ‘Why do you want to learn Surayt?’'] * 30
    key, subkey = rand.split(key)
    generated_sentences = generate(sentences, tokenizer, params, logits_processor, max_len=max_len, key=subkey)
    return generated_sentences

def main():
    top_k = 6
    max_len = 256

    params = load_params_from_disk()
    print('Successfully loaded model parameters!')

    key = rand.key(BEST_INTEGER, impl='rbg')
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    logits_processor = make_logits_processor(
        TopKSampler(top_k=top_k),
    )

    # Preheating
    for _ in range(5):
        generate_jax(params, tokenizer, logits_processor, max_len, key)

    # Benchmarking
    exec_time = timeit.timeit(lambda: generate_jax(params, tokenizer, logits_processor, max_len, key), number=10)
    print(f"Time taken for 10 executions of generate_jax: {exec_time:.5f} seconds")

if __name__ == '__main__':
    main()
