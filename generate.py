import argparse
import jax
import jax.numpy as jnp
import jax.random as rand
from transformers import LlamaTokenizer
from lib.generation import generate
from lib.LLM import Llama
from lib.logits_processing import PresencePenaltyProcessor, TopKSampler, TopPSampler, make_logits_processor
from lib.param_utils import load_params
from lib.multihost_utils import shard_model_params
from lib.seeding import BEST_INTEGER
import jax_smi

def load_params_from_disk(pickle_file: str) -> Llama:
    cpu_device = jax.devices('cpu')[0]
    with jax.default_device(cpu_device):
        params = load_params(pickle_file)
        params = jax.tree_map(lambda x: x.astype(jnp.float16), params)
    params = shard_model_params(params)
    return params

def main(pickle_file):
    top_k = 6
    # top_p = 0.05
    max_len = 256

    
    print('CONNECTED TPUs:', jax.device_count())
    print('LOCAL_DEVICES:', jax.local_devices())
    
    params = load_params_from_disk(pickle_file)
    print('Successfully loaded model parameters!')

    key = rand.key(BEST_INTEGER, impl='rbg')
    tokenizer = LlamaTokenizer.from_pretrained('meta-llama/Llama-2-70b-chat-hf', padding_side='left')
    tokenizer.pad_token = tokenizer.eos_token

    logits_processor = make_logits_processor(
        PresencePenaltyProcessor(penalty=0.05),
        TopKSampler(top_k=top_k),
        # TopPSampler(top_p=top_p),
    )

    batched_sentences = [
        [
            'Four score and seven years ago our fathers',
            'Marta is a Syriac student. Aday asks her: ‘Why do you want to learn Surayt?’',
        ],
        [
            'Marta is a Syriac student. Aday asks her: ‘Why do you want to learn Surayt?’',
            'Four score and seven years ago our fathers',
        ]
    ]

    for sentences in batched_sentences:
        key, subkey = rand.split(key)
        generated_sentences = generate(sentences, tokenizer, params, logits_processor, max_len=max_len, key=subkey)
        for sentence in generated_sentences:
            print(sentence, end='\n\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate text with Llama model.')
    parser.add_argument('pickle_file', type=str, help='The pickle file to load parameters from.')
    args = parser.parse_args()
    main(args.pickle_file)
