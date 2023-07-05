# from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_tpu; initialise_tpu('v4-16', n_devices=1, rank=0)

import jax.numpy as jnp
import jax.random as rand
from jax_smi import initialise_tracking
from transformers import LlamaTokenizer

# from lib.generation import TopKGenerationConfig, top_k
from lib.generation import TopPGenerationConfig, top_p
from lib.model import config_7B
from lib.param_utils import load_params
from lib.seeding import BEST_INTEGER

tokenizer = LlamaTokenizer.from_pretrained('../llama-weights')
tokenizer.pad_token = tokenizer.eos_token  # TODO: verify this
sentences = [
    'I believe the meaning of life is',
    'Simply put, the theory of relativity states that',
    'Thus, leveraging the potential of quantum computing, we can optimize complex algorithms, paving the way for breakthroughs in fields ranging from cryptography to molecular modeling',
]

def main() -> None:
    initialise_tracking()

    key = rand.PRNGKey(BEST_INTEGER)
    params = load_params('7B.pickle')
    # top_k_config = TopKGenerationConfig(eos_token_id=tokenizer.eos_token_id, max_length=128, top_k=10)
    top_p_config = TopPGenerationConfig(eos_token_id=tokenizer.eos_token_id, max_length=128, top_p=0.9)

    inputs = tokenizer(sentences, max_length=top_p_config.max_length, padding='max_length', return_tensors='jax')
    seq = inputs.input_ids.astype(jnp.uint16)
    attn_mask = inputs.attention_mask.astype(jnp.bool_)

    key, subkey = rand.split(key)
    # generated_seq = top_k(params, seq, attn_mask, key=subkey, model_config=config_7B, top_k_config=top_k_config)
    generated_seq = top_p(params, seq, attn_mask, key=subkey, model_config=config_7B, top_p_config=top_p_config)
    decoded_texts = tokenizer.batch_decode(generated_seq, skip_special_tokens=True)

    for decoded_text in decoded_texts:
        print(decoded_text, end='\n\n')

if __name__ == '__main__':
    main()
