# from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_tpu; initialise_tpu('v4-16', n_devices=4, rank=0)
# from lib.proc_init_utils import initialise_gpu; initialise_gpu()

import jax
import jax.numpy as jnp
import jax.random as rand
from jax.sharding import PositionalSharding
from jax_smi import initialise_tracking
from transformers import LlamaTokenizer

# from lib.generation import TopKGenerationConfig, top_k
from lib.generation import TopPGenerationConfig, top_p
from lib.model import model_config_llama2_7B
from lib.model.llama import create_model_parallel_sharding_llama
from lib.param_utils import load_params
from lib.seeding import BEST_INTEGER

tokenizer = LlamaTokenizer.from_pretrained('../llama-weights/Llama-2-7b-hf')
tokenizer.pad_token = tokenizer.eos_token  # TODO: verify this
sentences = [
    'I believe the meaning of life is',
    'Simply put, the theory of relativity states that',
    'Thus, leveraging the potential of quantum computing, we can optimize complex algorithms, paving the way for breakthroughs in fields ranging from cryptography to molecular modeling',
]

def main() -> None:
    initialise_tracking()

    key = rand.PRNGKey(BEST_INTEGER)
    with jax.default_device(jax.devices('cpu')[0]):
        params = load_params('llama2-7B.pickle')

    sharding = PositionalSharding(jax.devices())
    sharding_llama = create_model_parallel_sharding_llama(sharding)
    params = jax.device_put(params, sharding_llama)
    # top_k_config = TopKGenerationConfig(eos_token_id=tokenizer.eos_token_id, max_length=128, top_k=10)
    top_p_config = TopPGenerationConfig(eos_token_id=tokenizer.eos_token_id, max_length=128, top_p=0.9)

    inputs = tokenizer(sentences, max_length=top_p_config.max_length, padding='max_length', return_tensors='jax')
    seq = inputs.input_ids.astype(jnp.uint16)
    attn_mask = inputs.attention_mask.astype(jnp.bool_)

    shard_all = lambda x: jax.tree_map(lambda i: jax.device_put(i, sharding.replicate((0,))), x)

    seq, attn_mask = shard_all((seq, attn_mask))

    key, subkey = rand.split(key)
    config_llama2_7B_ = model_config_llama2_7B._replace(dropout_rate=None)
    # generated_seq = top_k(params, seq, attn_mask, key=subkey, model_config=model_config_llama1_7B, top_k_config=top_k_config)
    generated_seq = top_p(params, seq, attn_mask, key=subkey, model_config=config_llama2_7B_, top_p_config=top_p_config)
    decoded_texts = tokenizer.batch_decode(generated_seq, skip_special_tokens=True)

    for decoded_text in decoded_texts:
        print(decoded_text, end='\n\n')

if __name__ == '__main__':
    main()
