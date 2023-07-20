# from pathlib import Path; import sys; sys.path.append(str(Path(__file__).resolve().parent.parent))
from lib.proc_init_utils import initialise_tpu; initialise_tpu('v4-16', n_devices=1, rank=0)

import jax.numpy as jnp
import jax.random as rand
from transformers import LlamaTokenizer
from functools import partial
from tqdm import tqdm

from lib.dataloader import LlamaDataLoader
# from lib.generation import TopKGenerationConfig, top_k
from lib.generation import TopPGenerationConfig, top_p
from lib.model import config_llama2_7B
from lib.param_utils import load_params
from lib.seeding import BEST_INTEGER, HASHED_BUDDHA
from lib.gsm_data import GSMDataset, TestData, gsm_collate_fn_test

tokenizer = LlamaTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
tokenizer.pad_token = tokenizer.eos_token  # TODO: verify this

def main() -> None:
    key = rand.PRNGKey(BEST_INTEGER)
    params = load_params('upbeat-hill-35.pickle')
    max_len = 640
    batch_size = 32
    seed = HASHED_BUDDHA
    # top_k_config = TopKGenerationConfig(eos_token_id=tokenizer.eos_token_id, max_length=640, top_k=64)

    tokenizer = LlamaTokenizer.from_pretrained('NousResearch/Llama-2-7b-hf')
    top_p_config = TopPGenerationConfig(eos_token_id=tokenizer.eos_token_id, max_length=640, top_p=0.9)
    dataset = GSMDataset(split='test')
    collate_fn = partial(gsm_collate_fn_test, tokenizer, max_len)
    dataloader = LlamaDataLoader(dataset, collate_fn, batch_size, seed)
    config_llama2_7B_ = config_llama2_7B._replace(dropout_rate=None)

    pbar = tqdm(total=len(dataloader))
    with open('results.txt', 'w', encoding='utf-8') as f:
        for seq, seq_mask, labels in dataloader:
            key, subkey = rand.split(key)
            generated_seq = top_p(params, seq, seq_mask, key=subkey, model_config=config_llama2_7B_, top_p_config=top_p_config)
            decoded_texts = tokenizer.batch_decode(generated_seq, skip_special_tokens=True)

            for decoded_text in decoded_texts:
                print(decoded_text, labels, sep='\t', file=f)
            pbar.update()

if __name__ == '__main__':
    main()
